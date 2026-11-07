import argparse
import csv
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import config as cfg
from experiment import extract_patch
from models.mamba_ipf.dataset import TerrainPatchDataset, prepare_windows
from models.mamba_ipf.inference import resolve_device
from models.mamba_ipf.networks import build_model, load_pretrained_ipf, save_checkpoint

# Preparación de datos y entrenamiento de las dos instancias de 3DMambaIPF.
#   python train_mamba.py prepare                      -> ventanas de entrenamiento en data/mamba_windows/
#   python train_mamba.py train --task classifier      -> checkpoints/mamba_classifier.pt (fase 1)
#   python train_mamba.py train --task regressor       -> checkpoints/mamba_regressor.pt (fase 2)
# Ambas instancias se entrenan por separado y en cualquier orden: el regresor no usa las predicciones del
# clasificador, sino una fracción residual de outliers que simula sus fallos (MAMBA_RESIDUAL_OUTLIER_KEEP).

# --- Preparación ----------------------------------------------------------------------------------
def prepare(args):
    from data_loader import load_las_data   # laspy solo se necesita para preparar los datos

    # Caja XY del patch de evaluación de main.py: ninguna ventana que la toque entra en train/val
    eval_points, _, _, _ = load_las_data(cfg.FILE_PATH)
    _, _, (cx, cy) = extract_patch(eval_points, cfg.PATCH_SIZE, cfg.MIN_POINTS, cfg.MAX_ATTEMPTS, cfg.PATCH_SEED)
    half = cfg.PATCH_SIZE / 2
    exclude_box = (cx - half, cy - half, cx + half, cy + half)
    del eval_points
    print(f"Patch de evaluación excluido: {exclude_box}")

    # Se regeneran desde cero para no mezclar ventanas de configuraciones anteriores
    for old in cfg.MAMBA_WINDOWS_DIR.glob("*/*.npz"):
        old.unlink()

    for i, terrain in enumerate(cfg.MAMBA_TRAIN_TERRAINS):
        path = cfg.DATA_DIR / terrain / "points.laz"
        print(f"\n[{terrain}] cargando {path} ...")
        points, _, _, _ = load_las_data(path)
        saved, n_test = prepare_windows(
            terrain, points, cfg.MAMBA_WINDOWS_DIR, cfg.PATCH_SIZE, cfg.MIN_POINTS, cfg.MAMBA_WINDOWS_PER_SPLIT,
            cfg.MAMBA_VAL_FRACTION, cfg.MAMBA_TEST_FRACTION, exclude_box, seed=cfg.SEED + i,
        )
        print(f"[{terrain}] ventanas guardadas: train={saved['train']}, val={saved['val']} | reservadas para test: {n_test}")

# --- Pérdidas -------------------------------------------------------------------------------------
def classifier_loss(logits, labels):
    # BCE con peso para la clase minoritaria (los outliers suelen ser el 5-10% de los puntos)
    positives = labels.sum()
    pos_weight = ((labels.numel() - positives) / positives).clamp(1.0, 50.0) if positives > 0 else logits.new_tensor(1.0)
    return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)

def nearest_target(x, target):
    # Punto del objetivo más cercano a cada punto de x: (B, N, 3)
    nearest = torch.cdist(x, target).argmin(dim=-1)
    return torch.gather(target, 1, nearest.unsqueeze(-1).expand(-1, -1, 3))

def regressor_loss(inputs, displacements, clean, noise_std, noise_decay):
    # Pérdida supervisada de IterativePFN / 3DMambaIPF: cada módulo predice el desplazamiento hacia el
    # punto limpio más cercano; con currículo, los módulos intermedios apuntan a un objetivo con ruido
    # decreciente. Los puntos se ponderan por cercanía a la semilla (el borde tiene vecindario incompleto).
    # Se omite la pérdida de renderizado diferenciable del original: allí se calcula sobre copias numpy
    # desacopladas del grafo (render_views hace .detach().cpu().numpy()) y no aporta gradiente.
    seed_dist_sq = (inputs[0] ** 2).sum(dim=-1)
    seed_dist_sq = seed_dist_sq / (seed_dist_sq[:, -1:] / 9).clamp_min(1e-12)
    weights = torch.exp(-seed_dist_sq)
    weights = weights / weights.sum(dim=1, keepdim=True)

    total, std = 0.0, noise_std
    for i, (x, displacement) in enumerate(zip(inputs, displacements)):
        if noise_decay != 1 and i < len(inputs) - 1:
            std = std / noise_decay
            target = clean + torch.randn_like(clean) * std.view(-1, 1, 1)
        else:
            target = clean
        error = ((displacement - (nearest_target(x, target) - x)) ** 2).sum(dim=-1)
        total = total + (weights * error).sum(dim=1).mean()
    return total

# --- Entrenamiento --------------------------------------------------------------------------------
def make_dataset(task, split, items, augment, seed, args):
    files = sorted((cfg.MAMBA_WINDOWS_DIR / split).glob("*.npz"))
    noise_params = {
        "sigma": cfg.NOISE_SIGMA, "laplace_scale": cfg.NOISE_LAPLACE_SCALE, "axes": cfg.NOISE_AXES,
        "outlier_ratio": cfg.OUTLIER_RATIO, "scale_range": cfg.MAMBA_NOISE_SCALE_RANGE,
    }
    return TerrainPatchDataset(
        files, task, cfg.MAMBA_TRAIN_COMBINATIONS, noise_params, args.patch_size, args.samples_per_item,
        items, cfg.MAMBA_PATCH_RATIO, cfg.MAMBA_RESIDUAL_OUTLIER_KEEP, augment=augment, seed=seed,
    )

def flatten(batch, device):
    # (ítems, parches, ...) -> (ítems × parches, ...)
    return {k: v.reshape((-1,) + v.shape[2:]).to(device) for k, v in batch.items()}

def run_epoch(model, loader, task, device, noise_decay, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "batches": 0, "tp": 0, "fp": 0, "fn": 0, "err_before": 0.0, "err_after": 0.0, "points": 0}
    with torch.set_grad_enabled(training):
        for batch in loader:
            batch = flatten(batch, device)
            if task == "classifier":
                logits = model(batch["noisy"])
                loss = classifier_loss(logits, batch["labels"])
                predicted, labels = logits > 0, batch["labels"] > 0.5
                totals["tp"] += int((predicted & labels).sum())
                totals["fp"] += int((predicted & ~labels).sum())
                totals["fn"] += int((~predicted & labels).sum())
            else:
                inputs, displacements = model(batch["noisy"])
                loss = regressor_loss(inputs, displacements, batch["clean"], batch["noise_std"], noise_decay)
                # Distancia media (m) al punto limpio más cercano, antes y después de la red
                scale = batch["scale"].view(-1, 1)
                denoised = inputs[-1] + displacements[-1]
                before = (nearest_target(inputs[0], batch["clean"]) - inputs[0]).norm(dim=-1) * scale
                after = (nearest_target(denoised, batch["clean"]) - denoised).norm(dim=-1) * scale
                totals["err_before"] += float(before.sum())
                totals["err_after"] += float(after.sum())
                totals["points"] += before.numel()

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            totals["loss"] += float(loss)
            totals["batches"] += 1

    metrics = {"loss": totals["loss"] / max(totals["batches"], 1)}
    if task == "classifier":
        precision = totals["tp"] / max(totals["tp"] + totals["fp"], 1)
        recall = totals["tp"] / max(totals["tp"] + totals["fn"], 1)
        metrics.update(precision=precision, recall=recall, f1=2 * precision * recall / max(precision + recall, 1e-12))
    else:
        metrics.update(err_before_m=totals["err_before"] / max(totals["points"], 1),
                       err_after_m=totals["err_after"] / max(totals["points"], 1))
    return metrics

def train(args):
    device = resolve_device(args.device)
    if device == "cpu":
        print("AVISO: entrenando en CPU con la implementación de referencia de Mamba (muy lento). "
              "Úsalo solo para pruebas; el entrenamiento real requiere GPU y mamba_ssm.")
    torch.manual_seed(cfg.SEED)

    model_args = {"k_neighbors": args.k_neighbors}
    if args.task == "regressor":
        model_args["num_modules"] = args.num_modules
    model = build_model(args.task, model_args, device, args.impl)
    print(f"Modelo {args.task}: {sum(p.numel() for p in model.parameters()):,} parámetros en {device}")
    if args.init_from:
        loaded = load_pretrained_ipf(model, args.init_from)
        print(f"Inicializado desde {args.init_from}: {len(loaded)} tensores cargados")

    train_set = make_dataset(args.task, "train", args.items_per_epoch, True, cfg.SEED, args)
    val_set = make_dataset(args.task, "val", args.val_items, False, cfg.SEED + 1, args)
    loader_kwargs = {"batch_size": args.batch_items, "num_workers": args.num_workers, "pin_memory": device == "cuda"}
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=cfg.MAMBA_WEIGHT_DECAY)
    # Criterio de selección: F1 del clasificador (máximo) o error medio tras el regresor (mínimo)
    monitor, mode = ("f1", "max") if args.task == "classifier" else ("err_after_m", "min")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, patience=10, factor=0.5, min_lr=1e-7)

    output = args.output or (cfg.MAMBA_CLASSIFIER_CKPT if args.task == "classifier" else cfg.MAMBA_REGRESSOR_CKPT)
    log_path = output.with_suffix(".csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    best = None
    with open(log_path, "w", newline="", encoding="utf-8") as log_file:
        log = None
        for epoch in range(1, args.epochs + 1):
            start = time.time()
            train_set.set_epoch(epoch)
            train_metrics = run_epoch(model, train_loader, args.task, device, args.noise_decay, optimizer)
            val_metrics = run_epoch(model, val_loader, args.task, device, args.noise_decay)
            scheduler.step(val_metrics[monitor])

            row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "seconds": round(time.time() - start, 1)}
            row.update({f"train_{k}": v for k, v in train_metrics.items()})
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
            if log is None:
                log = csv.DictWriter(log_file, fieldnames=list(row))
                log.writeheader()
            log.writerow(row)
            log_file.flush()
            print(" | ".join(f"{k}={v:.5f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))

            value = val_metrics[monitor]
            if best is None or (value > best if mode == "max" else value < best):
                best = value
                save_checkpoint(output, model, model_args, args.patch_size,
                                extra={"epoch": epoch, "val": val_metrics, "monitor": monitor})
                print(f"  -> mejor {monitor}={value:.5f}, guardado en {output}")
    print(f"Entrenamiento terminado. Registro en {log_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Datos y entrenamiento de 3DMambaIPF para el pipeline de dos fases")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare", help="Corta las escenas de MAMBA_TRAIN_TERRAINS en ventanas train/val")

    t = sub.add_parser("train", help="Entrena el clasificador (fase 1) o el regresor (fase 2)")
    t.add_argument("--task", choices=["classifier", "regressor"], required=True)
    t.add_argument("--device", default=cfg.MAMBA_DEVICE)
    t.add_argument("--impl", default=cfg.MAMBA_IMPL)
    t.add_argument("--epochs", type=int, default=cfg.MAMBA_EPOCHS)
    t.add_argument("--lr", type=float, default=cfg.MAMBA_LR)
    t.add_argument("--patch_size", type=int, default=cfg.MAMBA_PATCH_SIZE)
    t.add_argument("--k_neighbors", type=int, default=cfg.MAMBA_K_NEIGHBORS)
    t.add_argument("--num_modules", type=int, default=cfg.MAMBA_NUM_MODULES)
    t.add_argument("--noise_decay", type=float, default=cfg.MAMBA_NOISE_DECAY)
    t.add_argument("--samples_per_item", type=int, default=cfg.MAMBA_SAMPLES_PER_ITEM)
    t.add_argument("--items_per_epoch", type=int, default=cfg.MAMBA_TRAIN_ITEMS_PER_EPOCH)
    t.add_argument("--val_items", type=int, default=cfg.MAMBA_VAL_ITEMS)
    t.add_argument("--batch_items", type=int, default=cfg.MAMBA_TRAIN_BATCH_ITEMS)
    t.add_argument("--num_workers", type=int, default=cfg.MAMBA_NUM_WORKERS)
    t.add_argument("--init_from", default=None, help="Checkpoint del 3DMambaIPF original para inicializar")
    t.add_argument("--output", type=Path, default=None)

    args = parser.parse_args()
    prepare(args) if args.command == "prepare" else train(args)
