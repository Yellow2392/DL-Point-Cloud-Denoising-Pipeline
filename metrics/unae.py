import numpy as np
import open3d as o3d
from metrics.neighbors import knn

def estimate_normals(points, k_neighbors):
    # Normales locales por PCA sobre los k vecinos más cercanos
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors))
    return np.asarray(pcd.normals)

def unae(pred, ref, k_neighbors):
    # Unoriented Normal Angle Error (grados): ángulo medio entre la normal de cada punto y la del
    # punto de referencia más cercano, ignorando el signo de la normal. Mide la consistencia de pendientes.
    pred_normals = estimate_normals(pred, k_neighbors)
    ref_normals = estimate_normals(ref, k_neighbors)

    idx, _ = knn(pred, ref)
    cos = np.abs(np.sum(pred_normals * ref_normals[idx], axis=1))
    return np.degrees(np.arccos(np.clip(cos, 0.0, 1.0))).mean()
