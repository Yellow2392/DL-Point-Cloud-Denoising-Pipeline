import numpy as np
from metrics.neighbors import knn

def hausdorff_distance(pred, ref, percentile=100.0):
    # Distancia de Hausdorff simétrica (en metros): peor desviación entre ambas nubes.
    # Con percentile < 100 se obtiene la variante robusta (p. ej. HD95), menos dominada por un único punto.
    _, d_pred_ref = knn(pred, ref)
    _, d_ref_pred = knn(ref, pred)
    return max(np.percentile(d_pred_ref, percentile), np.percentile(d_ref_pred, percentile))
