import numpy as np
from metrics.neighbors import knn

def compute_mean_distances(points, k_neighbors):
    # Distancia media de cada punto a sus k vecinos más cercanos (búsqueda vectorizada con Open3D).
    # Se piden k + 1 vecinos porque el primero es el propio punto (distancia 0).
    _, dist = knn(points, points, k=k_neighbors + 1)
    return dist[:, 1:].mean(axis=1)

def identify_outliers(mean_distances, std_mul):
    # Umbral global: media + std_mul × desviación estándar de las distancias medias de toda la nube
    threshold = mean_distances.mean() + std_mul * mean_distances.std()
    return mean_distances > threshold, threshold

def statistical_outlier_removal(points, k_neighbors, std_mul):
    # Fase 1 del pipeline: elimina los puntos cuyo vecindario es anómalamente disperso respecto del resto.
    # Devuelve los puntos conservados, la máscara de inliers sobre la nube de entrada (para saber qué se
    # eliminó y compararlo con los outliers inyectados) y el umbral de distancia aplicado.
    mean_distances = compute_mean_distances(points, k_neighbors)
    is_outlier, threshold = identify_outliers(mean_distances, std_mul)
    inlier_mask = ~is_outlier
    return points[inlier_mask], inlier_mask, threshold
