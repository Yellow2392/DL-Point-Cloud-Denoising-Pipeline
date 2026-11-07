import numpy as np
import torch
from .networks import load_checkpoint
from .patching import predict_patchwise

# Interfaz de las dos fases para el pipeline, con la misma forma que SOR y MLS:
# reciben y devuelven arrays numpy en coordenadas métricas originales.

_MODEL_CACHE = {}

def resolve_device(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device

def _load(checkpoint_path, device, mamba_impl):
    # Los modelos se cargan una sola vez y se reutilizan entre experimentos
    key = (str(checkpoint_path), device, mamba_impl)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = load_checkpoint(checkpoint_path, device, mamba_impl)
    return _MODEL_CACHE[key]

def mamba_outlier_removal(points, checkpoint_path, threshold=0.5, seed_k=6, batch_size=8, device="auto", mamba_impl="auto"):
    # Fase 1 (poda): probabilidad de outlier por punto y eliminación de los que superan el umbral.
    # Devuelve los puntos conservados, la máscara de inliers, las probabilidades y la cobertura de los parches.
    device = resolve_device(device)
    model, checkpoint = _load(checkpoint_path, device, mamba_impl)
    logits, _, covered = predict_patchwise(
        lambda x: model(x).unsqueeze(-1), points, checkpoint["patch_size"], seed_k, batch_size, device
    )
    probabilities = 1.0 / (1.0 + np.exp(-logits[:, 0]))
    probabilities[~covered] = 0.0   # sin predicción: se conserva el punto
    inlier_mask = probabilities <= threshold
    return points[inlier_mask], inlier_mask, probabilities, covered

def mamba_displacement(points, checkpoint_path, seed_k=6, batch_size=8, niters=1, device="auto", mamba_impl="auto"):
    # Fase 2 (suavizado): desplazamiento 3D por punto, aplicado niters veces sobre la nube completa.
    # Conserva la cardinalidad y el orden de los puntos.
    device = resolve_device(device)
    model, checkpoint = _load(checkpoint_path, device, mamba_impl)
    denoised = points.copy()
    covered = np.ones(len(points), dtype=bool)
    for _ in range(niters):
        displacement, scale, covered = predict_patchwise(
            lambda x: model.denoise(x) - x, denoised, checkpoint["patch_size"], seed_k, batch_size, device
        )
        # El desplazamiento sale en unidades normalizadas del parche: se devuelve a metros
        denoised[covered] += displacement[covered] * scale[covered, None]
    return denoised, covered
