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

NOISE_COMPONENTS = ("gaussian", "laplacian", "global_outliers", "clustered_outliers")

def component_seed(seed, component):
    # Semilla independiente por componente: un mismo componente genera idéntica perturbación en todas
    # las combinaciones (resultados comparables) y componentes distintos no comparten la secuencia aleatoria.
    return np.random.SeedSequence([seed, NOISE_COMPONENTS.index(component)])

def generate_outliers(reference, outlier_type, n_outliers, seed):
    # Los límites se toman de la nube de referencia (inliers), no de outliers previamente inyectados
    rng = np.random.default_rng(seed)
    outliers = np.zeros((n_outliers, 3))
    
    xyz_min = reference.min(axis=0)
    xyz_max = reference.max(axis=0)
    
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
    
    print(f"Outliers inyectados ({outlier_type}): {n_outliers} puntos")
    return outliers

def degrade_patch(patch_clean, components, sigma, laplace_scale, axes_mode, outlier_ratio, seed):
    # Aplica una combinación de degradaciones: primero el ruido posicional (los componentes se suman)
    # y luego se añaden los outliers al final del array.
    # Devuelve la nube degradada, la máscara de outliers y el desplazamiento de los puntos originales.
    unknown = set(components) - set(NOISE_COMPONENTS)
    if not components or unknown:
        raise ValueError(f"Componentes de ruido inválidos: {sorted(unknown) or 'lista vacía'}. "
                         f"Opciones: {NOISE_COMPONENTS}")
    
    patch_noisy = patch_clean.copy()
    if "gaussian" in components:
        patch_noisy, _, _ = apply_gaussian_noise(patch_noisy, sigma, axes_mode, component_seed(seed, "gaussian"))
    if "laplacian" in components:
        patch_noisy, _, _ = apply_laplacian_noise(patch_noisy, laplace_scale, axes_mode, component_seed(seed, "laplacian"))
    delta = patch_noisy - patch_clean
    
    # El número de outliers se calcula sobre el patch limpio para que no dependa de otros componentes
    n_outliers = int(len(patch_clean) * outlier_ratio)
    outlier_blocks = [
        generate_outliers(patch_noisy, outlier_type, n_outliers, component_seed(seed, f"{outlier_type}_outliers"))
        for outlier_type in ("global", "clustered")
        if f"{outlier_type}_outliers" in components
    ]
    patch_degraded = np.vstack([patch_noisy] + outlier_blocks)
    
    # Máscara booleana para identificar los outliers inyectados
    is_outlier = np.zeros(len(patch_degraded), dtype=bool)
    is_outlier[len(patch_noisy):] = True
    
    return patch_degraded, is_outlier, delta

def experiment_name(components):
    # Identificador para nombres de archivo y tablas, p. ej. "gaussian+clustered_outliers"
    return "+".join(components)

def describe_degradation(components, sigma, laplace_scale, outlier_ratio):
    # Etiqueta legible de la degradación aplicada, para títulos de gráficos
    labels = {
        "gaussian": f"Gaussiano σ={sigma:.3f} m",
        "laplacian": f"Laplaciano b={laplace_scale:.3f} m",
        "global_outliers": f"Outliers globales {outlier_ratio:.0%}",
        "clustered_outliers": f"Outliers agrupados {outlier_ratio:.0%}",
    }
    return " + ".join(labels[c] for c in components)
