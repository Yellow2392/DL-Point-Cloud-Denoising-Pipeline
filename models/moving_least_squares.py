import numpy as np
import open3d as o3d

def moving_least_squares_1st_order(points, k_neighbors, h_param):
    # Proyecta iterativamente los puntos hacia un plano tangente local.
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Usar KDTree para búsquedas eficientes de vecinos
    tree = o3d.geometry.KDTreeFlann(pcd)
    
    denoised_points = np.zeros_like(points)
    
    for i, p in enumerate(points):
        # 1. Encontrar los k vecinos más cercanos
        k, idx, dist2 = tree.search_knn_vector_3d(p, k_neighbors)
        
        # Si no hay suficientes puntos para definir un plano, conservar el punto original
        if k < 3:
            denoised_points[i] = p
            continue
            
        neighbors = points[idx]
        
        # 2. Calcular pesos mediante una función Gaussiana
        # dist2 ya contiene la distancia euclidiana al cuadrado
        weights = np.exp(-np.asarray(dist2) / (h_param ** 2))
        weights_sum = np.sum(weights)
        
        if weights_sum == 0:
            denoised_points[i] = p
            continue
            
        # 3. Calcular el centroide ponderado del vecindario local
        centroid = np.sum(neighbors * weights[:, np.newaxis], axis=0) / weights_sum
        
        # 4. Calcular la matriz de covarianza ponderada local
        diff = neighbors - centroid
        cov = (weights[:, np.newaxis] * diff).T @ diff
        
        # 5. Extraer la normal (vector propio asociado al menor valor propio)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        normal = eigenvectors[:, 0]
        
        # 6. Proyectar el punto ruidoso sobre el plano tangente estimado
        vector_to_centroid = p - centroid
        dist_to_plane = np.dot(vector_to_centroid, normal)
        denoised_points[i] = p - dist_to_plane * normal
        
    return denoised_points