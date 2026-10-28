import numpy as np

def extract_patch(points, patch_size, min_points, max_attempts, seed):
    rng = np.random.default_rng(seed)
    n = len(points)
    half = patch_size / 2
    candidate_indices = rng.choice(n, size=min(max_attempts, n), replace=False)
    
    for candidate_idx in candidate_indices:
        center_x, center_y = points[candidate_idx, :2]
        mask = (np.abs(points[:, 0] - center_x) <= half) & (np.abs(points[:, 1] - center_y) <= half)
        indices = np.flatnonzero(mask)
        
        if len(indices) >= min_points:
            return points[indices].copy(), indices, (center_x, center_y)
            
    raise ValueError(f"No se encontró un patch de {patch_size} m con al menos {min_points} puntos.")

def apply_gaussian_noise(patch_clean, sigma, axes_mode, seed):
    rng = np.random.default_rng(seed)
    patch_noisy = patch_clean.copy()
    
    if axes_mode == "xyz":
        noise = rng.normal(0.0, sigma, size=patch_clean.shape)
        patch_noisy += noise
    elif axes_mode == "z":
        noise = rng.normal(0.0, sigma, size=len(patch_clean))
        patch_noisy[:, 2] += noise
    else:
        raise ValueError("NOISE_AXES debe ser 'xyz' o 'z'.")
        
    delta = patch_noisy - patch_clean
    delta_norm = np.linalg.norm(delta, axis=1)
    
    print(f"Sigma configurado: {sigma:.4f} m")
    print(f"Desplazamiento 3D medio: {delta_norm.mean():.4f} m")
    
    return patch_noisy, delta, delta_norm

def apply_laplacian_noise(patch_clean, scale, axes_mode, seed):
    rng = np.random.default_rng(seed)
    patch_noisy = patch_clean.copy()
    
    if axes_mode == "xyz":
        noise = rng.laplace(0.0, scale, size=patch_clean.shape)
        patch_noisy += noise
    elif axes_mode == "z":
        noise = rng.laplace(0.0, scale, size=len(patch_clean))
        patch_noisy[:, 2] += noise
    else:
        raise ValueError("NOISE_AXES debe ser 'xyz' o 'z'.")
        
    delta = patch_noisy - patch_clean
    delta_norm = np.linalg.norm(delta, axis=1)
    
    print(f"Escala Laplaciana configurada (b): {scale:.4f} m")
    print(f"Desplazamiento 3D medio: {delta_norm.mean():.4f} m")
    
    return patch_noisy, delta, delta_norm

def inject_outliers(patch, outlier_type, ratio, seed):
    rng = np.random.default_rng(seed)
    n_outliers = int(len(patch) * ratio)
    outliers = np.zeros((n_outliers, 3))
    
    xyz_min = patch.min(axis=0)
    xyz_max = patch.max(axis=0)
    
    if outlier_type == "global":
        # Dispersos aleatoriamente en el espacio volumétrico
        outliers[:, 0] = rng.uniform(xyz_min[0], xyz_max[0], n_outliers)
        outliers[:, 1] = rng.uniform(xyz_min[1], xyz_max[1], n_outliers)
        outliers[:, 2] = rng.uniform(xyz_min[2], xyz_max[2] + 15.0, n_outliers)
    elif outlier_type == "clustered":
        # Grupos densos flotando sobre el terreno
        center = rng.uniform(xyz_min, xyz_max)
        center[2] = xyz_max[2] + rng.uniform(5.0, 15.0) 
        outliers = rng.normal(loc=center, scale=1.5, size=(n_outliers, 3))
    else:
        raise ValueError("outlier_type debe ser 'global' o 'clustered'")
        
    patch_with_outliers = np.vstack((patch, outliers))
    
    # Máscara booleana para identificar los outliers inyectados
    is_outlier = np.zeros(len(patch_with_outliers), dtype=bool)
    is_outlier[-n_outliers:] = True
    
    print(f"Outliers inyectados ({outlier_type}): {n_outliers} puntos")
    return patch_with_outliers, is_outlier