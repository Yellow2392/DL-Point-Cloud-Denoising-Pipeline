import numpy as np
import config as cfg
from data_loader import load_las_data, print_metadata
from stats import calculate_spatial_extent, calculate_density, print_stats, calculate_nearest_neighbor
from visualization import render_3d_heatmap, plot_profile_statistics, plot_patch_diagnostics
from experiment import extract_patch, apply_gaussian_noise

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
    print("\n" + "=" * 70 + "\nPATCH LOCAL + RUIDO GAUSSIANO\n" + "=" * 70)
    patch_clean, patch_indices, center = extract_patch(points, cfg.PATCH_SIZE, cfg.MIN_POINTS, cfg.MAX_ATTEMPTS, cfg.PATCH_SEED)
    patch_noisy, delta, delta_norm = apply_gaussian_noise(patch_clean, cfg.NOISE_SIGMA, cfg.NOISE_AXES, cfg.NOISE_SEED)
    
    # Preparar visualización del parche
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

if __name__ == "__main__":
    main()