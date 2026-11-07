import numpy as np
import torch

# Utilidades de parches compartidas por el entrenamiento y la inferencia, para que ambos vean
# exactamente la misma representación de entrada:
#   - Parche = los K vecinos 3D más cercanos de una semilla, ORDENADOS por distancia a ella
#     (Mamba procesa la secuencia en ese orden; la semilla es el primer punto).
#   - Normalización: centrado en la semilla y escalado por el radio horizontal (XY) del parche.
#     Se usa el radio XY y no el 3D para que los outliers que flotan sobre el terreno no inflen la escala.

def normalize_patch(patch, seed_point):
    centered = patch - seed_point
    scale = np.sqrt((centered[:, :2] ** 2).sum(axis=1).max())
    scale = max(scale, 1e-6)
    return (centered / scale).astype(np.float32), scale

def knn_from_seed(points, seed_point, k, prefilter=6):
    # Índices de los k vecinos 3D de seed_point, ordenados por distancia. Prefiltra por distancia XY
    # (barata) y verifica la exactitud: un punto a distancia 3D <= r también está a distancia XY <= r,
    # así que el resultado es exacto si el radio 3D del parche no supera el radio XY prefiltrado.
    k = min(k, len(points))
    d_xy = ((points[:, :2] - seed_point[:2]) ** 2).sum(axis=1)
    m = min(len(points), prefilter * k)
    candidates = np.argpartition(d_xy, m - 1)[:m] if m < len(points) else np.arange(len(points))
    d_3d = ((points[candidates] - seed_point) ** 2).sum(axis=1)
    order = np.argsort(d_3d, kind="stable")[:k]
    if m < len(points) and d_3d[order[-1]] > d_xy[candidates].max():
        d_3d = ((points - seed_point) ** 2).sum(axis=1)
        return np.argsort(d_3d, kind="stable")[:k]
    return candidates[order]

def farthest_point_sampling(points, n_samples, device, first_index=0):
    # Semillas repartidas uniformemente por la nube (para que los parches la cubran por completo)
    pts = torch.as_tensor(points, dtype=torch.float32, device=device)
    n_samples = min(n_samples, len(pts))
    selected = torch.empty(n_samples, dtype=torch.long, device=device)
    min_dist = torch.full((len(pts),), float("inf"), device=device)
    current = first_index
    for i in range(n_samples):
        selected[i] = current
        min_dist = torch.minimum(min_dist, ((pts - pts[current]) ** 2).sum(dim=1))
        current = int(torch.argmax(min_dist))
    return selected.cpu().numpy()

def knn_patches(points, seed_indices, patch_size, device, chunk=64):
    # Para cada semilla, índices (S, K) de sus K vecinos ordenados por distancia y distancias (S, K)
    pts = torch.as_tensor(points, dtype=torch.float32, device=device)
    all_idx, all_dist = [], []
    for start in range(0, len(seed_indices), chunk):
        seeds = pts[torch.as_tensor(seed_indices[start:start + chunk], device=device)]
        dist, idx = torch.cdist(seeds, pts).topk(patch_size, dim=1, largest=False, sorted=True)
        all_idx.append(idx.cpu().numpy())
        all_dist.append(dist.cpu().numpy())
    return np.concatenate(all_idx), np.concatenate(all_dist)

def stitch_best(values, patch_idx, patch_dist, n_points):
    # Cada punto toma el valor predicho en el parche donde está más cerca de la semilla (relativo al
    # radio del parche), como el "patch stitching" de IterativePFN. Devuelve también qué puntos se cubrieron.
    rel_dist = patch_dist / np.maximum(patch_dist[:, -1:], 1e-12)
    flat_idx = patch_idx.reshape(-1)
    flat_dist = rel_dist.reshape(-1)
    flat_values = values.reshape(len(flat_idx), -1)

    order = np.lexsort((flat_dist, flat_idx))
    sorted_idx = flat_idx[order]
    first = np.r_[True, sorted_idx[1:] != sorted_idx[:-1]]

    stitched = np.zeros((n_points, flat_values.shape[1]), dtype=flat_values.dtype)
    covered = np.zeros(n_points, dtype=bool)
    stitched[sorted_idx[first]] = flat_values[order[first]]
    covered[sorted_idx[first]] = True
    return stitched, covered

@torch.no_grad()
def predict_patchwise(model_fn, points, patch_size, seed_k, batch_size, device):
    # Recorre la nube completa con parches solapados, aplica model_fn a lotes de parches normalizados y
    # cose el resultado. model_fn: (B, K, 3) normalizado -> (B, K, D).
    # Devuelve las salidas por punto (N, D), la escala de normalización de cada una y la cobertura.
    origin = points.mean(axis=0)
    local = points - origin   # en float64: tras centrar, float32 conserva precisión submilimétrica
    n_points = len(local)
    patch_size = min(patch_size, n_points)
    n_seeds = max(1, int(np.ceil(seed_k * n_points / patch_size)))

    seed_indices = farthest_point_sampling(local, n_seeds, device)
    patch_idx, patch_dist = knn_patches(local, seed_indices, patch_size, device)

    outputs, scales = [], []
    for start in range(0, len(seed_indices), batch_size):
        batch, batch_scales = [], []
        for s in range(start, min(start + batch_size, len(seed_indices))):
            normalized, scale = normalize_patch(local[patch_idx[s]], local[seed_indices[s]])
            batch.append(normalized)
            batch_scales.append(scale)
        out = model_fn(torch.from_numpy(np.stack(batch)).to(device))
        outputs.append(out.float().cpu().numpy())
        scales.extend(batch_scales)

    outputs = np.concatenate(outputs)                                     # (S, K, D)
    scales = np.broadcast_to(np.asarray(scales)[:, None, None], outputs.shape[:2] + (1,))
    stitched, covered = stitch_best(np.concatenate((outputs, scales), axis=-1), patch_idx, patch_dist, n_points)
    return stitched[:, :-1], stitched[:, -1], covered
