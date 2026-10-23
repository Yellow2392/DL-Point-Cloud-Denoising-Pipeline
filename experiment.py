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