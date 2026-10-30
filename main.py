import numpy as np
import config as cfg
from data_loader import load_las_data, print_metadata
from stats import calculate_spatial_extent, calculate_density, print_stats, calculate_nearest_neighbor
from visualization import render_3d_heatmap, plot_profile_statistics, plot_patch_diagnostics, plot_denoising_comparison
from experiment import extract_patch, apply_gaussian_noise, apply_laplacian_noise, inject_outliers
from models.moving_least_squares import moving_least_squares_1st_order

def main():
    # 1. Carga de datos
    points, intensity, classification, header = load_las_data(cfg.FILE_PATH)
    n = len(points)
    print_metadata(header, n, cfg.FILE_PATH)
    """
    # 2. Análisis espacial y estadístico
    extent, area_km2 = calculate_spatial_extent(points)
    calculate_density(n, area_km2)
    
    print("\n" + "=" * 70 + "\nESTADÍSTICAS\n" + "=" * 70)
    print_stats("Elevación Z (m)", points[:, 2])
    print_stats("Intensidad", intensity)
    
    classes, counts = np.unique(classification, return_counts=True)
    nn_distances, sample_size = calculate_nearest_neighbor(points, cfg.NN_SAMPLE, cfg.SEED)
    print_stats("Distancia al vecino más cercano en la muestra (m)", nn_distances)
    
    # 3. Visualización y gráficos
    render_3d_heatmap(points, cfg.VIS_POINTS, cfg.SEED, f"{cfg.TERRAIN} - Elevación")
    stats_plot_path = cfg.OUTPUT_DIR / "profile_statistics.png"
    plot_profile_statistics(points, intensity, classes, counts, nn_distances, sample_size, cfg.TERRAIN, stats_plot_path)
    """
    # 4. Experimento: Patch y Ruido Gaussiano
    print("\n" + "=" * 70 + "\nPATCH LOCAL + INYECCIÓN DE RUIDO\n" + "=" * 70)
    patch_clean, patch_indices, center = extract_patch(points, cfg.PATCH_SIZE, cfg.MIN_POINTS, cfg.MAX_ATTEMPTS, cfg.PATCH_SEED)
    
    # Aplicar ruido a elección (Gaussiano o Laplaciano)
    # patch_noisy, delta, delta_norm = apply_gaussian_noise(patch_clean, cfg.NOISE_SIGMA, cfg.NOISE_AXES, cfg.NOISE_SEED)
    patch_noisy, delta, delta_norm = apply_laplacian_noise(patch_clean, cfg.NOISE_LAPLACE_SCALE, cfg.NOISE_AXES, cfg.NOISE_SEED)
    
    # Inyectar outliers al patch ruidoso
    patch_corrupted, is_outlier = inject_outliers(patch_noisy, cfg.OUTLIER_TYPE, cfg.OUTLIER_RATIO, cfg.NOISE_SEED)

    # 5. Denoising: Aplicación de Moving Least Squares
    print("\n" + "=" * 70 + "\nDENOISING: MOVING LEAST SQUARES (MLS)\n" + "=" * 70)
    print(f"Iniciando MLS con {cfg.MLS_K_NEIGHBORS} vecinos y h={cfg.MLS_H_PARAM}...")
    
    # Aplicamos MLS a la nube corrompida
    patch_denoised = moving_least_squares_1st_order(
        patch_corrupted, 
        k_neighbors=cfg.MLS_K_NEIGHBORS, 
        h_param=cfg.MLS_H_PARAM
    )
    
    # Calculamos la distancia RMSE global (simplificada) para reportar en consola
    # Evaluamos solo sobre los puntos verdaderos para ver si los reubicó correctamente
    # (los outliers severos en MLS requerirían métodos estadísticos adicionales previos)
    valid_points_mask = ~is_outlier
    rmse_before = np.sqrt(np.mean(np.sum((patch_corrupted[valid_points_mask] - patch_clean)**2, axis=1)))
    rmse_after = np.sqrt(np.mean(np.sum((patch_denoised[valid_points_mask] - patch_clean)**2, axis=1)))
    
    print(f"RMSE inicial (solo ruido): {rmse_before:.4f} m")
    print(f"RMSE tras MLS: {rmse_after:.4f} m")
    
    # Generar visualización del denoising
    plot_denoising_path = cfg.OUTPUT_DIR / "denoising_mls_comparison.png"

    # Submuestreo para graficar sin colapsar la memoria/vista
    n_vis = min(cfg.MAX_DISPLAY, len(patch_clean))
    idx_vis = np.random.default_rng(123).choice(len(patch_clean), n_vis, replace=False)
    
    # Asumiremos visualizar las mismas proporciones
    plot_denoising_comparison(
        patch_clean[idx_vis], 
        patch_corrupted[idx_vis], 
        patch_denoised[idx_vis], 
        plot_denoising_path
    )

    """
    # Preparar visualización (Asegurar que delta solo aplique a los puntos no-espurios en visualization.py)
    n_vis = min(cfg.MAX_DISPLAY, len(patch_clean))
    idx_vis = np.random.default_rng(123).choice(len(patch_clean), n_vis, replace=False)
    
    plot_path = cfg.OUTPUT_DIR / "patch_gaussian_diagnostic.png"
    plot_patch_diagnostics(
        patch_clean[idx_vis], patch_noisy[idx_vis], delta, delta[idx_vis], 
        cfg.NOISE_AXES, cfg.NOISE_SIGMA, cfg.NOISE_SEED, cfg.AMPLIFY, plot_path
    )
    
    # Guardar artefactos de experimento
    data_path = cfg.OUTPUT_DIR / "patch_gaussian_data.npz"
    np.savez_compressed(
        data_path, clean=patch_clean, noisy=patch_noisy, source_indices=patch_indices,
        sigma=np.array(cfg.NOISE_SIGMA), axes=np.array(cfg.NOISE_AXES), patch_size=np.array(cfg.PATCH_SIZE)
    )
    print(f"Ejecución finalizada. Datos guardados en {cfg.OUTPUT_DIR}")
    """

if __name__ == "__main__":
    main()