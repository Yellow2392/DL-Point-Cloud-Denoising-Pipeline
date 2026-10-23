import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def render_3d_heatmap(points, vis_points, seed, window_name):
    rng = np.random.default_rng(seed)
    n = len(points)
    vis_size = min(vis_points, n)
    vis_indices = rng.choice(n, vis_size, replace=False)
    
    points_vis = points[vis_indices]
    z_vis = points_vis[:, 2]
    
    z_low, z_high = np.percentile(z_vis, 2), np.percentile(z_vis, 98)
    z_norm = np.clip((z_vis - z_low) / (z_high - z_low + 1e-12), 0, 1)
    colors = plt.get_cmap("turbo")(z_norm)[:, :3]
    
    pcd_vis = o3d.geometry.PointCloud()
    pcd_vis.points = o3d.utility.Vector3dVector(points_vis)
    pcd_vis.colors = o3d.utility.Vector3dVector(colors)
    
    print("\nControles 3D: Arrastrar (rotar), Rueda (zoom), Shift+Arrastrar (trasladar), Q/ESC (cerrar)")
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1400, height=900)
    vis.add_geometry(pcd_vis)
    
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    
    vis.run()
    vis.destroy_window()

def plot_profile_statistics(points, intensity, classes, counts, nn_distances, sample_size, terrain, output_path):
    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Perfil estadístico - {terrain}\n{len(points):,} puntos", fontsize=15)
    
    axs[0, 0].hist(points[:, 2], bins=150)
    axs[0, 0].set_title("Distribución de elevación (Z)")
    
    axs[0, 1].hist(intensity, bins=150)
    axs[0, 1].set_title("Distribución de intensidad")
    
    axs[1, 0].bar(classes.astype(str), counts)
    axs[1, 0].set_title("Distribución de clasificación")
    
    axs[1, 1].hist(nn_distances, bins=100)
    axs[1, 1].set_title(f"Distancia al vecino más cercano\n(muestra de {sample_size:,})")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()

def plot_patch_diagnostics(clean_vis, noisy_vis, delta, delta_vis, axes_mode, sigma, noise_seed, amplify, output_path):
    z_min = min(clean_vis[:, 2].min(), noisy_vis[:, 2].min())
    z_max = max(clean_vis[:, 2].max(), noisy_vis[:, 2].max())
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    sc1 = axs[0].scatter(clean_vis[:, 0], clean_vis[:, 1], c=clean_vis[:, 2], s=2, cmap="turbo", vmin=z_min, vmax=z_max, rasterized=True)
    axs[0].set_title("Patch limpio")
    axs[0].set_aspect("equal", adjustable="box")
    
    axs[1].scatter(noisy_vis[:, 0], noisy_vis[:, 1], c=noisy_vis[:, 2], s=2, cmap="turbo", vmin=z_min, vmax=z_max, rasterized=True)
    axs[1].set_title(f"Patch con ruido gaussiano ({axes_mode})")
    axs[1].set_aspect("equal", adjustable="box")
    
    fig.colorbar(sc1, ax=axs[:2], label="Elevación Z (m)", shrink=0.8)
    
    if axes_mode == "xyz":
        for p, d in zip(clean_vis, delta_vis):
            axs[2].plot([p[0], p[0] + amplify * d[0]], [p[1], p[1] + amplify * d[1]], linewidth=0.5, alpha=0.5)
        axs[2].scatter(clean_vis[:, 0], clean_vis[:, 1], s=3, label="Posición limpia")
        axs[2].scatter(clean_vis[:, 0] + amplify * delta_vis[:, 0], clean_vis[:, 1] + amplify * delta_vis[:, 1], s=3, label=f"Desplazamiento XY x{amplify}")
        axs[2].set_title("Desplazamiento horizontal amplificado")
        axs[2].set_aspect("equal", adjustable="datalim")
        axs[2].legend(markerscale=2, fontsize=8)
    else:
        axs[2].hist(delta[:, 2], bins=60)
        axs[2].axvline(0, linestyle="--")
        axs[2].set_title("Distribución del ruido vertical")
    
    fig.suptitle(f"sigma={sigma:.3f} m | semilla={noise_seed}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.show()
    plt.close(fig)