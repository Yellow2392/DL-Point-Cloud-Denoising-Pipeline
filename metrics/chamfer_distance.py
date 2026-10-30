import numpy as np
from metrics.neighbors import knn

def chamfer_distance(pred, ref):
    # Distancia de Chamfer simétrica (en metros): promedio de las distancias medias al vecino
    # más cercano en ambos sentidos.
    _, d_pred_ref = knn(pred, ref)
    _, d_ref_pred = knn(ref, pred)
    return 0.5 * (d_pred_ref.mean() + d_ref_pred.mean())

def _dcd_term(source, target, alpha):
    # Para cada x en source: 1 - exp(-alpha * ||x - y||) / n_y, donde y es su vecino más cercano
    # en target y n_y es cuántos puntos de source comparten ese mismo vecino.
    idx, dist = knn(source, target)
    n_y = np.bincount(idx, minlength=len(target))[idx]
    return np.mean(1.0 - np.exp(-alpha * dist) / n_y)

def density_aware_chamfer_distance(pred, ref, alpha):
    # Density-aware Chamfer Distance (Wu et al., 2021). Acotada en [0, 1]; penaliza además
    # la distribución no uniforme (aglomeración de puntos o huecos), no solo la distancia.
    # alpha (1/m) controla la sensibilidad a la distancia.
    return 0.5 * (_dcd_term(pred, ref, alpha) + _dcd_term(ref, pred, alpha))
