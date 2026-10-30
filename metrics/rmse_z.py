import numpy as np
from metrics.neighbors import knn

def rmse_z(pred, ref, k=4):
    # Error altimétrico frente a la superficie de referencia.
    # La cota de referencia en la posición XY de cada punto se interpola (IDW) a partir de sus
    # k vecinos más cercanos en el plano XY, por lo que no requiere correspondencia 1 a 1.
    pred_xy = np.column_stack((pred[:, :2], np.zeros(len(pred))))
    ref_xy = np.column_stack((ref[:, :2], np.zeros(len(ref))))

    idx, dist = knn(pred_xy, ref_xy, k)
    weights = 1.0 / (dist + 1e-9)
    z_ref = np.sum(weights * ref[idx, 2], axis=1) / np.sum(weights, axis=1)

    return np.sqrt(np.mean((pred[:, 2] - z_ref) ** 2))
