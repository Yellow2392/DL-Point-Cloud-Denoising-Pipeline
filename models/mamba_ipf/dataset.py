import numpy as np
import torch
from torch.utils.data import Dataset
from experiment import degrade_patch
from .patching import knn_from_seed, normalize_patch

# Datos de entrenamiento a partir de las escenas LiDAR del proyecto.
# 1) prepare_windows: corta cada escena en ventanas de WINDOW_SIZE m (como los patches de 100 m del
#    pipeline) y las reparte en train / val / test por ventana completa (partición espacial, sin fuga
#    entre conjuntos). Las ventanas que tocan el patch de evaluación de main.py se reservan siempre para test.
# 2) TerrainPatchDataset: en cada muestra degrada una ventana completa con degrade_patch (la misma
#    función y los mismos componentes que los experimentos) y extrae varios parches kNN de ella.

def tile_windows(points, window_size, min_points):
    # Índices de los puntos de cada ventana de la rejilla XY: {(ix, iy): índices}
    cells = np.floor((points[:, :2] - points[:, :2].min(axis=0)) / window_size).astype(np.int64)
    keys = cells[:, 0] * (cells[:, 1].max() + 1) + cells[:, 1]
    order = np.argsort(keys, kind="stable")
    unique_keys, starts, counts = np.unique(keys[order], return_index=True, return_counts=True)
    n_cols = cells[:, 1].max() + 1
    return {
        (int(key // n_cols), int(key % n_cols)): order[start:start + count]
        for key, start, count in zip(unique_keys, starts, counts)
        if count >= min_points
    }

def window_intersects(points, box):
    # box: (xmin, ymin, xmax, ymax)
    xy_min, xy_max = points[:, :2].min(axis=0), points[:, :2].max(axis=0)
    return not (xy_max[0] < box[0] or xy_min[0] > box[2] or xy_max[1] < box[1] or xy_min[1] > box[3])

def prepare_windows(terrain, points, output_dir, window_size, min_points, windows_per_split, val_fraction,
                    test_fraction, exclude_box, seed):
    # Guarda cada ventana como npz con coordenadas relativas a su centro (float32) y el centro (float64)
    rng = np.random.default_rng(seed)
    windows = tile_windows(points, window_size, min_points)
    keys = list(windows)
    rng.shuffle(keys)

    # Ventanas incompletas (bordes de la escena) se descartan: menos de la mitad de la densidad típica
    typical = np.median([len(idx) for idx in windows.values()])
    keys = [k for k in keys if len(windows[k]) >= 0.5 * typical]

    n_val = int(round(len(keys) * val_fraction))
    n_test = int(round(len(keys) * test_fraction))
    splits = {}
    for i, key in enumerate(keys):
        window_points = points[windows[key]]
        if exclude_box is not None and window_intersects(window_points, exclude_box):
            splits[key] = "test"
        elif i < n_val:
            splits[key] = "val"
        elif i < n_val + n_test:
            splits[key] = "test"
        else:
            splits[key] = "train"

    saved = {"train": 0, "val": 0, "test": 0}
    for key in keys:
        split = splits[key]
        if split == "test" or saved[split] >= windows_per_split.get(split, 0):
            continue   # test no se guarda: la evaluación final la hace main.py
        window_points = points[windows[key]]
        center = window_points.mean(axis=0)
        path = output_dir / split / f"{terrain}_{key[0]}_{key[1]}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, points=(window_points - center).astype(np.float32), center=center)
        saved[split] += 1
    return saved, sum(1 for s in splits.values() if s == "test")

def random_rotation_z(rng):
    # Aumentación compatible con terreno: rotación libre alrededor de Z y reflexión opcional en X.
    # No se rota en X/Y: la vertical tiene significado físico (outliers sobre/bajo el suelo).
    angle = rng.uniform(0, 2 * np.pi)
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    if rng.random() < 0.5:
        rotation[:, 0] *= -1
    return rotation

class TerrainPatchDataset(Dataset):
    # Cada ítem: una ventana degradada aleatoriamente y samples_per_item parches extraídos de ella.
    #   task="classifier": entrada = nube degradada completa; etiqueta = outlier inyectado (0/1) por punto.
    #   task="regressor":  entrada = nube degradada sin los outliers inyectados, salvo una fracción
    #                      residual_outlier_keep que simula los que la fase 1 no detecta; objetivo =
    #                      parche limpio vecino (pérdida por vecino más cercano, como 3DMambaIPF).

    def __init__(self, window_files, task, combinations, noise_params, patch_size, samples_per_item,
                 items_per_epoch, patch_ratio=1.2, residual_outlier_keep=0.1, augment=True, seed=0):
        super().__init__()
        if task not in ("classifier", "regressor"):
            raise ValueError("task debe ser 'classifier' o 'regressor'.")
        self.windows = [np.load(f)["points"].astype(np.float64) for f in window_files]
        if not self.windows:
            raise ValueError("No hay ventanas: ejecuta primero 'python train_mamba.py prepare'.")
        self.task = task
        self.combinations = combinations
        self.noise_params = noise_params
        self.patch_size = patch_size
        self.samples_per_item = samples_per_item
        self.items_per_epoch = items_per_epoch
        self.patch_ratio = patch_ratio
        self.residual_outlier_keep = residual_outlier_keep
        self.augment = augment
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        # Con DataLoader no persistente los workers se recrean en cada época y ven el nuevo valor
        self.epoch = epoch

    def __len__(self):
        return self.items_per_epoch

    def _degrade(self, clean, rng):
        p = self.noise_params
        components = self.combinations[rng.integers(len(self.combinations))]
        low, high = p["scale_range"]
        degraded, is_outlier, _ = degrade_patch(
            clean, components,
            sigma=p["sigma"] * rng.uniform(low, high),
            laplace_scale=p["laplace_scale"] * rng.uniform(low, high),
            axes_mode=p["axes"],
            outlier_ratio=p["outlier_ratio"] * rng.uniform(low, high),
            seed=int(rng.integers(2**31)),
            verbose=False,
        )
        return degraded, is_outlier

    def __getitem__(self, index):
        rng = np.random.default_rng([self.seed, self.epoch, index])
        clean = self.windows[rng.integers(len(self.windows))]
        degraded, is_outlier = self._degrade(clean, rng)

        # Índice de cada punto degradado en la nube limpia (los inliers conservan el orden de 'clean')
        source_idx = np.arange(len(degraded))
        if self.task == "regressor":
            keep = ~is_outlier | (rng.random(len(degraded)) < self.residual_outlier_keep)
            degraded, is_outlier, source_idx = degraded[keep], is_outlier[keep], source_idx[keep]

        noisy_patches, labels, clean_patches, noise_stds, scales = [], [], [], [], []
        for _ in range(self.samples_per_item):
            seed_point = degraded[rng.integers(len(degraded))]
            idx = knn_from_seed(degraded, seed_point, self.patch_size)
            noisy, scale = normalize_patch(degraded[idx], seed_point)
            scales.append(np.float32(scale))
            rotation = random_rotation_z(rng) if self.augment else np.eye(3, dtype=np.float32)
            noisy_patches.append(noisy @ rotation.T)

            if self.task == "classifier":
                labels.append(is_outlier[idx].astype(np.float32))
            else:
                clean_idx = knn_from_seed(clean, seed_point, int(self.patch_ratio * self.patch_size))
                clean_patch = ((clean[clean_idx] - seed_point) / scale).astype(np.float32)
                clean_patches.append(clean_patch @ rotation.T)
                # Magnitud del ruido del parche (unidades normalizadas), para el currículo de la pérdida
                inliers = idx[~is_outlier[idx]]
                residual = degraded[inliers] - clean[source_idx[inliers]]
                noise_stds.append(np.float32(np.sqrt(np.mean(residual ** 2)) / scale) if len(inliers) else np.float32(0.0))

        item = {"noisy": torch.from_numpy(np.stack(noisy_patches)), "scale": torch.tensor(scales)}
        if self.task == "classifier":
            item["labels"] = torch.from_numpy(np.stack(labels))
        else:
            item["clean"] = torch.from_numpy(np.stack(clean_patches))
            item["noise_std"] = torch.tensor(noise_stds)
        return item
