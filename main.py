import numpy as np
import config as cfg
from data_loader import load_las_data, print_metadata
from stats import calculate_spatial_extent, calculate_density, print_stats, calculate_nearest_neighbor
from visualization import render_3d_heatmap, plot_profile_statistics, plot_patch_diagnostics, plot_denoising_comparison, view_clean_vs_degraded
from experiment import extract_patch, degrade_patch, experiment_name, describe_degradation
from models.moving_least_squares import moving_least_squares_1st_order, estimate_bandwidth
from metrics.evaluation import evaluate, print_metrics_table, save_summary_csv

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
    # 4. Experimento: Patch local (común a todas las combinaciones de ruido)
    print("\n" + "=" * 70 + "\nPATCH LOCAL\n" + "=" * 70)
    patch_clean, patch_indices, center = extract_patch(points, cfg.PATCH_SIZE, cfg.MIN_POINTS, cfg.MAX_ATTEMPTS, cfg.PATCH_SEED)
    print(f"Patch de {cfg.PATCH_SIZE} m centrado en {center}: {len(patch_clean):,} puntos")

    all_results = {}
    for components in cfg.NOISE_COMBINATIONS:
        all_results[experiment_name(components)] = run_experiment(patch_clean, patch_indices, components)

    # 7. Resumen de todas las combinaciones
    print("\n" + "=" * 70 + "\nRESUMEN DE EXPERIMENTOS\n" + "=" * 70)
    print_metrics_table({
        f"{name} | {cloud}": metrics
        for name, results in all_results.items()
        for cloud, metrics in results.items()
    })
    summary_path = cfg.OUTPUT_DIR / "metrics_summary.csv"
    save_summary_csv(all_results, summary_path)
    print(f"\nResumen guardado en {summary_path}")

def run_experiment(patch_clean, patch_indices, components):
    name = experiment_name(components)
    noise_label = describe_degradation(components, cfg.NOISE_SIGMA, cfg.NOISE_LAPLACE_SCALE, cfg.OUTLIER_RATIO)
    print("\n" + "#" * 70 + f"\nEXPERIMENTO: {name}\n{noise_label}\n" + "#" * 70)

    patch_corrupted, is_outlier, delta = degrade_patch(
        patch_clean, components, cfg.NOISE_SIGMA, cfg.NOISE_LAPLACE_SCALE,
        cfg.NOISE_AXES, cfg.OUTLIER_RATIO, cfg.NOISE_SEED
    )

    # 5. Denoising: Aplicación de Moving Least Squares
    print("\n" + "=" * 70 + "\nDENOISING: MOVING LEAST SQUARES (MLS)\n" + "=" * 70)
    # El ancho de banda se estima sobre la nube corrompida: la referencia limpia no está disponible en la práctica
    h_param = cfg.MLS_H_PARAM
    if h_param is None:
        h_param, spacing = estimate_bandwidth(patch_corrupted, cfg.MLS_H_FACTOR)
        print(f"Espaciado mediano entre vecinos: {spacing:.4f} m -> h = {cfg.MLS_H_FACTOR} × espaciado")
    print(f"Iniciando MLS con {cfg.MLS_K_NEIGHBORS} vecinos y h={h_param:.4f} m...")

    # Aplicamos MLS a la nube corrompida
    patch_denoised = moving_least_squares_1st_order(
        patch_corrupted,
        k_neighbors=cfg.MLS_K_NEIGHBORS,
        h_param=h_param
    )

    # 6. Métricas frente a la nube de referencia limpia
    # Las filas "(inliers)" excluyen los outliers inyectados: MLS no elimina puntos, solo los proyecta,
    # así que comparar ambas variantes separa el efecto del suavizado del de los outliers remanentes.
    print("\n" + "=" * 70 + "\nMÉTRICAS FRENTE A LA REFERENCIA\n" + "=" * 70)
    clouds = {"Corrompida": patch_corrupted, "MLS": patch_denoised}
    if is_outlier.any():
        valid_points_mask = ~is_outlier
        clouds["Corrompida (inliers)"] = patch_corrupted[valid_points_mask]
        clouds["MLS (inliers)"] = patch_denoised[valid_points_mask]
    results = {
        cloud_name: evaluate(cloud, patch_clean, cfg.DCD_ALPHA, cfg.NORMAL_K_NEIGHBORS)
        for cloud_name, cloud in clouds.items()
    }
    print_metrics_table(results)

    # Generar visualización del denoising
    plot_denoising_path = cfg.OUTPUT_DIR / f"denoising_mls_comparison_{name}.png"

    # Submuestreo para graficar sin colapsar la memoria/vista.
    # Las nubes corrompida y restaurada se muestrean sobre todos sus puntos (outliers incluidos,
    # que están al final del array); la limpia, por separado sobre los suyos.
    rng_vis = np.random.default_rng(123)
    idx_clean_vis = rng_vis.choice(len(patch_clean), min(cfg.MAX_DISPLAY, len(patch_clean)), replace=False)
    idx_corrupted_vis = rng_vis.choice(len(patch_corrupted), min(cfg.MAX_DISPLAY, len(patch_corrupted)), replace=False)

    plot_denoising_comparison(
        patch_clean[idx_clean_vis],
        patch_corrupted[idx_corrupted_vis],
        patch_denoised[idx_corrupted_vis],
        is_outlier[idx_corrupted_vis],
        noise_label,
        plot_denoising_path,
        show=cfg.SHOW_FIGURES
    )

    # Visor 3D interactivo para inspeccionar las diferencias entre la nube limpia y la degradada
    if cfg.OPEN3D_VIEWER:
        view_clean_vs_degraded(
            patch_clean, patch_corrupted, is_outlier, patch_denoised,
            window_name=f"{cfg.TERRAIN} - {name}",
            max_points=cfg.VIEW_MAX_POINTS, max_lines=cfg.VIEW_MAX_LINES,
            error_amplify=cfg.VIEW_ERROR_AMPLIFY, seed=cfg.SEED
        )

    """
    # Preparar visualización (Asegurar que delta solo aplique a los puntos no-espurios en visualization.py)
    n_vis = min(cfg.MAX_DISPLAY, len(patch_clean))
    idx_vis = np.random.default_rng(123).choice(len(patch_clean), n_vis, replace=False)

    plot_path = cfg.OUTPUT_DIR / f"patch_{name}_diagnostic.png"
    plot_patch_diagnostics(
        patch_clean[idx_vis], patch_corrupted[idx_vis], delta, delta[idx_vis],
        cfg.NOISE_AXES, noise_label, cfg.NOISE_SEED, cfg.AMPLIFY, plot_path
    )

    # Guardar artefactos de experimento
    data_path = cfg.OUTPUT_DIR / f"patch_{name}_data.npz"
    np.savez_compressed(
        data_path, clean=patch_clean, degraded=patch_corrupted, is_outlier=is_outlier, source_indices=patch_indices,
        components=np.array(components), sigma=np.array(cfg.NOISE_SIGMA),
        laplace_scale=np.array(cfg.NOISE_LAPLACE_SCALE), outlier_ratio=np.array(cfg.OUTLIER_RATIO),
        axes=np.array(cfg.NOISE_AXES), patch_size=np.array(cfg.PATCH_SIZE)
    )
    print(f"Datos guardados en {cfg.OUTPUT_DIR}")
    """

    return results

if __name__ == "__main__":
    main()
