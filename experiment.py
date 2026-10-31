import numpy as np
import config as cfg

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

def generate_outliers(reference, outlier_type, n_outliers, seed, z_buffer_above=6.0, z_buffer_below=2.0, n_clusters=3):
    # Global: dispersos en la elevación relativa a la superficie
    # Clustered: cúmulos acumulados sobre puntos del terreno (como vegetación u obstáculos pequeños)
    if n_outliers <= 0:
        return np.empty((0, 3), dtype=np.float64)

    rng = np.random.default_rng(seed)
    n_ref = len(reference)
    outliers = np.zeros((n_outliers, 3), dtype=np.float64)

    if outlier_type == "global":
        # 1. Seleccionamos puntos base de la referencia para muestrear el plano XY y la cota Z local
        base_indices = rng.choice(n_ref, size=n_outliers, replace=True)
        base_points = reference[base_indices]

        # Pequeña dispersión en el plano horizontal (XY) dentro del entorno local
        outliers[:, :2] = base_points[:, :2] + rng.uniform(-1.5, 1.5, size=(n_outliers, 2))

        # 2. Desplazamiento vertical relativo al terreno:
        # La mayoría (80%) sobre el terreno (polvo, aves, aerosoles) y una fracción (20%) bajo el suelo (multipath)
        is_above = rng.random(n_outliers) > 0.20
        dz = np.where(
            is_above,
            rng.uniform(0.3, z_buffer_above, size=n_outliers),   # Sobre el suelo
            rng.uniform(-z_buffer_below, -0.3, size=n_outliers)  # Bajo el suelo (multipath)
        )
        outliers[:, 2] = base_points[:, 2] + dz

    elif outlier_type == "clustered":
        # Distribuir los outliers en k cúmulos pequeños sobre el terreno
        k = min(n_clusters, n_outliers)
        counts = np.full(k, n_outliers // k)
        counts[:n_outliers % k] += 1

        # Elegir k puntos base del terreno como centroides ancla
        anchor_indices = rng.choice(n_ref, size=k, replace=False)
        offset = 0

        for count, anchor_idx in zip(counts, anchor_indices):
            anchor = reference[anchor_idx]

            # El centroide flota a una altura realista sobre el terreno local (ej. 1.5 a 4.5 metros)
            cluster_center = np.array([
                anchor[0] + rng.uniform(-2.0, 2.0),
                anchor[1] + rng.uniform(-2.0, 2.0),
                anchor[2] + rng.uniform(1.5, 4.5)
            ])

            # Dispersión del clúster (simulando un arbusto o copa de árbol)
            # Desviación horizontal de ~0.8m y vertical de ~0.5m
            cluster_scale = np.array([0.8, 0.8, 0.5])
            cluster_pts = rng.normal(loc=cluster_center, scale=cluster_scale, size=(count, 3))

            outliers[offset:offset + count] = cluster_pts
            offset += count
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
        generate_outliers(
            reference=patch_noisy,
            outlier_type=outlier_type,
            n_outliers=n_outliers,
            seed=component_seed(seed, f"{outlier_type}_outliers"),
            z_buffer_above=cfg.OUTLIER_Z_ABOVE,
            z_buffer_below=cfg.OUTLIER_Z_BELOW,
            n_clusters=cfg.OUTLIER_CLUSTERS_COUNT
        )
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
