import importlib.util
import numpy as np
import config as cfg
from data_loader import load_las_data, print_metadata
from stats import calculate_spatial_extent, calculate_density, print_stats, calculate_nearest_neighbor
from visualization import render_3d_heatmap, plot_profile_statistics, plot_patch_diagnostics, plot_denoising_comparison, view_clean_vs_degraded
from experiment import extract_patch, degrade_patch, experiment_name, describe_degradation
from models.statistical_outlier_removal import statistical_outlier_removal
from models.moving_least_squares import moving_least_squares_1st_order, estimate_bandwidth
from metrics.evaluation import evaluate, print_metrics_table, save_summary_csv
from metrics.outlier_detection import outlier_detection_metrics

# Nombres de los métodos de cada fase en tablas y figuras
PHASE1_LABELS = {"sor": "SOR", "mamba": "Mamba-cls", "none": "Sin poda"}
PHASE2_LABELS = {"mls": "MLS", "mamba": "Mamba-reg", "none": "Sin suavizado"}

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

    pipelines = available_pipelines()
    all_results = {}
    all_detection = {}
    for components in cfg.NOISE_COMBINATIONS:
        name = experiment_name(components)
        all_results[name], all_detection[name] = run_experiment(patch_clean, patch_indices, components, pipelines)

    # 8. Resumen de todas las combinaciones
    print("\n" + "=" * 70 + "\nRESUMEN DE EXPERIMENTOS\n" + "=" * 70)
    print_metrics_table({
        f"{name} | {cloud}": metrics
        for name, results in all_results.items()
        for cloud, metrics in results.items()
    })
    summary_path = cfg.OUTPUT_DIR / "metrics_summary.csv"
    save_summary_csv(all_results, summary_path)
    print(f"\nResumen guardado en {summary_path}")

    all_detection = {name: detection for name, detection in all_detection.items() if detection}
    if all_detection:
        print("\n" + "=" * 70 + "\nRESUMEN FASE 1: DETECCIÓN DE OUTLIERS\n" + "=" * 70)
        print_metrics_table({
            f"{name} | {method}": metrics
            for name, detection in all_detection.items()
            for method, metrics in detection.items()
        })
        detection_path = cfg.OUTPUT_DIR / "outlier_removal_summary.csv"
        save_summary_csv(all_detection, detection_path)
        print(f"\nResumen guardado en {detection_path}")

def available_pipelines():
    # Descarta (con aviso) los pipelines que usan 3DMambaIPF si falta PyTorch o el checkpoint entrenado,
    # para que el resto de la evaluación funcione también en equipos sin el entorno de deep learning.
    checkpoints = {1: cfg.MAMBA_CLASSIFIER_CKPT, 2: cfg.MAMBA_REGRESSOR_CKPT}
    has_torch = importlib.util.find_spec("torch") is not None
    pipelines = []
    for phase1, phase2 in cfg.PIPELINES:
        missing = [
            f"checkpoint de la fase {phase} ({checkpoints[phase]})"
            for phase, method in ((1, phase1), (2, phase2))
            if method == "mamba" and not checkpoints[phase].exists()
        ]
        if "mamba" in (phase1, phase2) and not has_torch:
            missing.insert(0, "PyTorch")
        if missing:
            print(f"AVISO: se omite el pipeline {pipeline_label(phase1, phase2)}: falta {', '.join(missing)}")
        else:
            pipelines.append((phase1, phase2))
    return pipelines

def pipeline_label(phase1, phase2):
    return f"{PHASE1_LABELS[phase1]} + {PHASE2_LABELS[phase2]}"

def run_outlier_removal(method, points):
    # Fase 1: devuelve los puntos conservados y la máscara de inliers sobre la nube de entrada
    if method == "sor":
        kept_points, kept_mask, threshold = statistical_outlier_removal(
            points, k_neighbors=cfg.SOR_K_NEIGHBORS, std_mul=cfg.SOR_STD_MUL
        )
        print(f"SOR con {cfg.SOR_K_NEIGHBORS} vecinos y std_mul={cfg.SOR_STD_MUL}: umbral={threshold:.4f} m")
    elif method == "mamba":
        from models.mamba_ipf.inference import mamba_outlier_removal
        kept_points, kept_mask, _, covered = mamba_outlier_removal(
            points, cfg.MAMBA_CLASSIFIER_CKPT, threshold=cfg.MAMBA_OUTLIER_THRESHOLD, seed_k=cfg.MAMBA_SEED_K,
            batch_size=cfg.MAMBA_BATCH_SIZE, device=cfg.MAMBA_DEVICE, mamba_impl=cfg.MAMBA_IMPL
        )
        print(f"Clasificador Mamba (umbral {cfg.MAMBA_OUTLIER_THRESHOLD}); puntos sin parche: {np.sum(~covered):,}")
    elif method == "none":
        kept_points, kept_mask = points, np.ones(len(points), dtype=bool)
    else:
        raise ValueError(f"Método de la fase 1 desconocido: {method}")
    return kept_points, kept_mask

def run_displacement(method, points):
    # Fase 2: devuelve la nube con los puntos desplazados (misma cardinalidad y orden)
    if method == "mls":
        # El ancho de banda se estima sobre la entrada de MLS: la referencia limpia no está disponible en la práctica
        h_param = cfg.MLS_H_PARAM
        if h_param is None:
            h_param, spacing = estimate_bandwidth(points, cfg.MLS_H_FACTOR)
            print(f"Espaciado mediano entre vecinos: {spacing:.4f} m -> h = {cfg.MLS_H_FACTOR} × espaciado")
        print(f"Iniciando MLS con {cfg.MLS_K_NEIGHBORS} vecinos y h={h_param:.4f} m...")
        return moving_least_squares_1st_order(points, k_neighbors=cfg.MLS_K_NEIGHBORS, h_param=h_param)
    if method == "mamba":
        from models.mamba_ipf.inference import mamba_displacement
        denoised, covered = mamba_displacement(
            points, cfg.MAMBA_REGRESSOR_CKPT, seed_k=cfg.MAMBA_SEED_K, batch_size=cfg.MAMBA_BATCH_SIZE,
            niters=cfg.MAMBA_NITERS, device=cfg.MAMBA_DEVICE, mamba_impl=cfg.MAMBA_IMPL
        )
        print(f"Regresor Mamba ({cfg.MAMBA_NITERS} pasada/s); puntos sin parche: {np.sum(~covered):,}")
        return denoised
    if method == "none":
        return points.copy()
    raise ValueError(f"Método de la fase 2 desconocido: {method}")

def run_experiment(patch_clean, patch_indices, components, pipelines):
    name = experiment_name(components)
    noise_label = describe_degradation(components, cfg.NOISE_SIGMA, cfg.NOISE_LAPLACE_SCALE, cfg.OUTLIER_RATIO)
    print("\n" + "#" * 70 + f"\nEXPERIMENTO: {name}\n{noise_label}\n" + "#" * 70)

    patch_corrupted, is_outlier, delta = degrade_patch(
        patch_clean, components, cfg.NOISE_SIGMA, cfg.NOISE_LAPLACE_SCALE,
        cfg.NOISE_AXES, cfg.OUTLIER_RATIO, cfg.NOISE_SEED
    )

    # Las filas "(inliers)" excluyen los outliers inyectados (en el resultado final, los que la fase 1 no
    # eliminó), separando el efecto del suavizado del de los outliers remanentes.
    clouds = {"Corrompida": patch_corrupted}
    if is_outlier.any():
        clouds["Corrompida (inliers)"] = patch_corrupted[~is_outlier]
    detection = {}
    outputs = []

    for phase1, phase2 in pipelines:
        label = pipeline_label(phase1, phase2)

        # 5. Fase 1: poda de outliers
        print("\n" + "=" * 70 + f"\n{label} | FASE 1: {PHASE1_LABELS[phase1]}\n" + "=" * 70)
        patch_kept, kept_mask = run_outlier_removal(phase1, patch_corrupted)
        n_removed = len(patch_corrupted) - len(patch_kept)
        print(f"Puntos eliminados: {n_removed:,} de {len(patch_corrupted):,} ({n_removed / len(patch_corrupted):.2%})")
        if phase1 != "none":
            detection[PHASE1_LABELS[phase1]] = outlier_detection_metrics(~kept_mask, is_outlier)
            print("Detección frente a los outliers inyectados: " +
                  ", ".join(f"{k}={v:.4f}" for k, v in detection[PHASE1_LABELS[phase1]].items()))

        # 6. Fase 2: predicción de desplazamientos sobre los puntos conservados
        print("\n" + "=" * 70 + f"\n{label} | FASE 2: {PHASE2_LABELS[phase2]}\n" + "=" * 70)
        patch_denoised = run_displacement(phase2, patch_kept)
        # Outliers inyectados que sobrevivieron a la fase 1 (alineado con patch_kept y patch_denoised)
        denoised_is_outlier = is_outlier[kept_mask]

        clouds[PHASE1_LABELS[phase1]] = patch_kept
        clouds[label] = patch_denoised
        if is_outlier.any():
            clouds[f"{label} (inliers)"] = patch_denoised[~denoised_is_outlier]
        outputs.append((phase1, phase2, kept_mask, patch_denoised, denoised_is_outlier))

    # 7. Métricas frente a la nube de referencia limpia
    print("\n" + "=" * 70 + "\nMÉTRICAS FRENTE A LA REFERENCIA\n" + "=" * 70)
    results = {
        cloud_name: evaluate(cloud, patch_clean, cfg.DCD_ALPHA, cfg.NORMAL_K_NEIGHBORS)
        for cloud_name, cloud in clouds.items()
    }
    print_metrics_table(results)

    for phase1, phase2, kept_mask, patch_denoised, denoised_is_outlier in outputs:
        label = pipeline_label(phase1, phase2)

        # Generar visualización del denoising
        plot_denoising_path = cfg.OUTPUT_DIR / f"denoising_{phase1}_{phase2}_comparison_{name}.png"

        # Submuestreo para graficar sin colapsar la memoria/vista.
        # Cada nube se muestrea sobre todos sus puntos (outliers incluidos, que están al final del array).
        rng_vis = np.random.default_rng(123)
        idx_clean_vis = rng_vis.choice(len(patch_clean), min(cfg.MAX_DISPLAY, len(patch_clean)), replace=False)
        idx_corrupted_vis = rng_vis.choice(len(patch_corrupted), min(cfg.MAX_DISPLAY, len(patch_corrupted)), replace=False)
        idx_denoised_vis = rng_vis.choice(len(patch_denoised), min(cfg.MAX_DISPLAY, len(patch_denoised)), replace=False)

        plot_denoising_comparison(
            patch_clean[idx_clean_vis],
            patch_corrupted[idx_corrupted_vis],
            is_outlier[idx_corrupted_vis],
            patch_denoised[idx_denoised_vis],
            denoised_is_outlier[idx_denoised_vis],
            noise_label,
            plot_denoising_path,
            method_label=label,
            show=cfg.SHOW_FIGURES
        )

        # Visor 3D interactivo para inspeccionar las diferencias entre la nube limpia y la degradada.
        # El visor necesita la nube restaurada alineada con la degradada: los puntos eliminados en la
        # fase 1 conservan su posición degradada y se identifican con kept_mask.
        if cfg.OPEN3D_VIEWER:
            restored_aligned = patch_corrupted.copy()
            restored_aligned[kept_mask] = patch_denoised
            view_clean_vs_degraded(
                patch_clean, patch_corrupted, is_outlier, restored_aligned, kept_mask=kept_mask,
                window_name=f"{cfg.TERRAIN} - {name} - {label}",
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

    return results, detection

if __name__ == "__main__":
    main()
