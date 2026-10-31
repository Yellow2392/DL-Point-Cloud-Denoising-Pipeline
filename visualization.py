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

def plot_patch_diagnostics(clean_vis, noisy_vis, delta, delta_vis, axes_mode, noise_label, noise_seed, amplify, output_path):
    z_min = min(clean_vis[:, 2].min(), noisy_vis[:, 2].min())
    z_max = max(clean_vis[:, 2].max(), noisy_vis[:, 2].max())

    fig, axs = plt.subplots(
        1, 3,
        figsize=(18, 5.5),
        constrained_layout=True
    )

    # 1. Patch limpio
    sc1 = axs[0].scatter(
        clean_vis[:, 0], clean_vis[:, 1],
        c=clean_vis[:, 2], s=2, cmap="turbo",
        vmin=z_min, vmax=z_max, rasterized=True
    )
    axs[0].set_title("Patch limpio")
    axs[0].set_aspect("equal", adjustable="box")

    # 2. Patch con ruido
    axs[1].scatter(
        noisy_vis[:, 0], noisy_vis[:, 1],
        c=noisy_vis[:, 2], s=2, cmap="turbo",
        vmin=z_min, vmax=z_max, rasterized=True
    )
    axs[1].set_title(f"Patch con ruido {noise_label} ({axes_mode})")
    axs[1].set_aspect("equal", adjustable="box")

    # 3. Diagnóstico del ruido
    if axes_mode == "xyz":
        for p, d in zip(clean_vis, delta_vis):
            axs[2].plot(
                [p[0], p[0] + amplify * d[0]],
                [p[1], p[1] + amplify * d[1]],
                linewidth=0.5, alpha=0.5
            )

        axs[2].scatter(
            clean_vis[:, 0], clean_vis[:, 1],
            s=3, label="Posición limpia"
        )
        axs[2].scatter(
            clean_vis[:, 0] + amplify * delta_vis[:, 0],
            clean_vis[:, 1] + amplify * delta_vis[:, 1],
            s=3, label=f"Desplazamiento XY ×{amplify}"
        )
        axs[2].set_title("Desplazamiento horizontal amplificado")
        axs[2].set_aspect("equal", adjustable="datalim")
        axs[2].legend(markerscale=2, fontsize=8)
    else:
        axs[2].hist(delta[:, 2], bins=60)
        axs[2].axvline(0, linestyle="--")
        axs[2].set_title("Distribución del ruido vertical")
        axs[2].set_xlabel("Ruido en Z (m)")
        axs[2].set_ylabel("Frecuencia")

    # Barra de color compartida: debajo de los tres paneles
    cbar = fig.colorbar(
        sc1,
        ax=axs,
        location="bottom",
        pad=0.10,
        shrink=0.75,
        aspect=40
    )
    cbar.set_label("Elevación Z (m)")

    fig.suptitle(
        f"Diagnóstico del ruido | {noise_label} | semilla={noise_seed}"
    )

    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_denoising_comparison(clean_points, noisy_points, denoised_points, is_outlier, noise_label, output_path, show=True):
    # La escala de color se fija con la nube limpia: si incluyera los outliers (flotando varios metros
    # sobre el terreno) comprimiría el relieve en pocos colores. Los outliers se dibujan aparte.
    z_min = clean_points[:, 2].min()
    z_max = clean_points[:, 2].max()

    fig, axs = plt.subplots(
        1, 3,
        figsize=(18, 5.5),
        constrained_layout=True
    )

    titles = [
        "1. Original / Limpio",
        "2. Degradado",
        "3. Denoising: Moving Least Squares"
    ]

    no_outliers = np.zeros(len(clean_points), dtype=bool)
    datasets = [(clean_points, no_outliers), (noisy_points, is_outlier), (denoised_points, is_outlier)]

    for ax, (points, outlier_mask), title in zip(axs, datasets, titles):
        inliers = points[~outlier_mask]
        sc = ax.scatter(
            inliers[:, 0], inliers[:, 1],
            c=inliers[:, 2],
            s=2,
            cmap="turbo",
            vmin=z_min,
            vmax=z_max,
            rasterized=True
        )
        if outlier_mask.any():
            outliers = points[outlier_mask]
            ax.scatter(
                outliers[:, 0], outliers[:, 1],
                s=8, c="magenta", marker="x", linewidths=0.6,
                label=f"Outliers ({outlier_mask.sum():,} mostrados)",
                rasterized=True, zorder=3
            )
            ax.legend(loc="upper right", fontsize=8, markerscale=1.5)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="datalim")

    # Barra de color común
    cbar = fig.colorbar(
        sc,
        ax=axs,
        location="bottom",
        pad=0.10,
        shrink=0.75,
        aspect=40
    )
    cbar.set_label("Elevación Z (m)")

    fig.suptitle(
        f"Evaluación de Denoising: MLS Clásico | {noise_label}",
        fontsize=16
    )

    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


OUTLIER_COLOR = np.array([1.0, 0.0, 1.0])   # Magenta: no aparece en los mapas de color usados
CLEAN_GRAY = np.array([0.55, 0.55, 0.55])

class CleanVsDegradedViewer:
    # Visor Open3D interactivo para comparar la nube limpia con la degradada (y opcionalmente la restaurada).
    # Todas las nubes se submuestrean con los mismos índices, conservando la correspondencia punto a punto.

    def __init__(self, clean, degraded, is_outlier, denoised, window_name, max_points, max_lines, error_amplify, seed):
        rng = np.random.default_rng(seed)
        n_clean = len(clean)
        idx_inliers = np.sort(rng.choice(n_clean, min(max_points, n_clean), replace=False))
        # Los outliers se submuestrean en la misma proporción que los inliers
        idx_outliers = np.flatnonzero(is_outlier)
        n_outliers_vis = int(round(len(idx_outliers) * len(idx_inliers) / n_clean))
        idx_outliers = np.sort(rng.choice(idx_outliers, n_outliers_vis, replace=False))
        idx_lines = rng.choice(len(idx_inliers), min(max_lines, len(idx_inliers)), replace=False)

        # Open3D renderiza en float32: con coordenadas UTM (~10^6 m) la precisión cae a decímetros y el
        # ruido se confundiría con artefactos de cuantización. Se centra todo en el centroide del patch limpio.
        self.origin = clean.mean(axis=0)
        self.clean = clean[idx_inliers]
        self.degraded = np.vstack((degraded[idx_inliers], degraded[idx_outliers]))
        self.denoised = None if denoised is None else np.vstack((denoised[idx_inliers], denoised[idx_outliers]))
        self.n_outliers = len(idx_outliers)
        self.idx_lines = idx_lines

        # Error por punto respecto de su posición limpia (solo inliers); escala común para comparar nubes
        n_in = len(idx_inliers)
        self.degraded_error = np.linalg.norm(self.degraded[:n_in] - self.clean, axis=1)
        self.denoised_error = None if denoised is None else np.linalg.norm(self.denoised[:n_in] - self.clean, axis=1)
        self.error_vmax = np.percentile(self.degraded_error, 95) + 1e-12
        self.z_low, self.z_high = np.percentile(self.clean[:, 2], [2, 98])

        self.window_name = window_name
        self.error_amplify = error_amplify
        self.color_mode = "error"        # "error" o "elevation"
        self.side_by_side = False
        self.show = {"clean": True, "degraded": True, "denoised": False, "lines": False}
        self.vis = None

    # --- Construcción de geometrías ---
    def _amplified(self, points):
        # Exagera el desplazamiento de cada inlier respecto de su posición limpia sin deformar el relieve.
        # Los outliers no tienen posición limpia de referencia y se muestran sin modificar.
        n_in = len(self.clean)
        p = points.copy()
        p[:n_in] = self.clean + self.error_amplify * (points[:n_in] - self.clean)
        return p

    def _transform(self, points, slot):
        p = points - self.origin
        if self.side_by_side:
            # Paneles contiguos en X: limpia | degradada | restaurada
            p[:, 0] += slot * 1.1 * np.ptp(self.clean[:, 0])
        return p

    def _elevation_colors(self, points):
        z_norm = np.clip((points[:, 2] - self.z_low) / (self.z_high - self.z_low + 1e-12), 0, 1)
        return plt.get_cmap("turbo")(z_norm)[:, :3]

    def _error_colors(self, errors):
        return plt.get_cmap("inferno")(np.clip(errors / self.error_vmax, 0, 1))[:, :3]

    def _cloud_colors(self, points, errors):
        n_in = len(self.clean)
        if self.color_mode == "elevation":
            colors = self._elevation_colors(points)
        elif errors is None:
            colors = np.tile(CLEAN_GRAY, (len(points), 1))
        else:
            colors = np.vstack((self._error_colors(errors), np.zeros((len(points) - n_in, 3))))
        colors[n_in:] = OUTLIER_COLOR
        return colors

    def _point_cloud(self, points, colors, slot):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self._transform(points, slot))
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    def _displacement_lines(self):
        # Vectores limpio -> degradado coloreados por la magnitud del error
        start = self._transform(self.clean[self.idx_lines], 0)
        end = self._transform(self._amplified(self.degraded)[self.idx_lines], 0)
        n = len(self.idx_lines)
        lines = o3d.geometry.LineSet()
        lines.points = o3d.utility.Vector3dVector(np.vstack((start, end)))
        lines.lines = o3d.utility.Vector2iVector(np.column_stack((np.arange(n), np.arange(n) + n)))
        lines.colors = o3d.utility.Vector3dVector(self._error_colors(self.degraded_error[self.idx_lines]))
        return lines

    def geometries(self):
        geoms = []
        if self.show["clean"]:
            geoms.append(self._point_cloud(self.clean, self._cloud_colors(self.clean, None), 0))
        if self.show["degraded"]:
            degraded = self._amplified(self.degraded)
            geoms.append(self._point_cloud(degraded, self._cloud_colors(degraded, self.degraded_error), 1))
        if self.show["denoised"] and self.denoised is not None:
            denoised = self._amplified(self.denoised)
            geoms.append(self._point_cloud(denoised, self._cloud_colors(denoised, self.denoised_error), 2))
        if self.show["lines"] and not self.side_by_side:
            geoms.append(self._displacement_lines())
        return geoms

    # --- Interacción ---
    def refresh(self, reset_view=False):
        # Solo se reencuadra la cámara al cambiar la disposición; al mostrar u ocultar capas o
        # amplificar el error se conserva el punto de vista para comparar en el mismo encuadre.
        self.vis.clear_geometries()
        for geom in self.geometries():
            self.vis.add_geometry(geom, reset_bounding_box=False)
        if reset_view:
            self.vis.reset_view_point(True)
        return True

    def _toggle(self, key):
        def callback(vis):
            self.show[key] = not self.show[key]
            print(f"  {key}: {'visible' if self.show[key] else 'oculta'}")
            return self.refresh()
        return callback

    def _toggle_color_mode(self, vis):
        self.color_mode = "elevation" if self.color_mode == "error" else "error"
        print(f"  Color: {self.color_mode}")
        return self.refresh()

    def _toggle_layout(self, vis):
        self.side_by_side = not self.side_by_side
        print(f"  Disposición: {'lado a lado' if self.side_by_side else 'superpuesta'}")
        return self.refresh(reset_view=True)

    def _scale_error(self, factor):
        def callback(vis):
            self.error_amplify = float(np.clip(self.error_amplify * factor, 1.0, 256.0))
            print(f"  Amplificación del error: ×{self.error_amplify:g}")
            return self.refresh()
        return callback

    def print_help(self):
        print("\n" + "=" * 70 + f"\nVISOR 3D: {self.window_name}\n" + "=" * 70)
        print(f"Puntos mostrados: {len(self.clean):,} del patch, {self.n_outliers:,} outliers (magenta)")
        print(f"Escala de error (inferno): 0 -> {self.error_vmax:.3f} m (p95 del error de la nube degradada)")
        print("Teclas:")
        print("  1 / 2 / 3 : mostrar/ocultar nube limpia (gris) / degradada / restaurada por MLS")
        print("  V         : mostrar/ocultar vectores de desplazamiento limpio -> degradado")
        print("  C         : alternar color por error respecto de la limpia / por elevación")
        print("  M         : alternar disposición superpuesta / lado a lado")
        print("  E / D     : aumentar / reducir la amplificación del error (×2) en degradada, restaurada y vectores")
        print("  + / -     : tamaño de punto;  R: reencuadrar;  Q/ESC: cerrar y continuar")
        print("Sugerencias:")
        print("  - Superpuesta + color por error: la nube gris es la referencia y los puntos amarillos son")
        print("    los más desplazados. Oculta la limpia (1) para ver solo el patrón del error.")
        print("  - Amplificación (E) con vista lateral rasante: un ruido de 0.1-0.3 m no se ve en un patch de")
        print("    100 m; con ×8-×16 la nube degradada se separa como una 'nube gruesa' alrededor de la gris.")
        print("    El color sigue indicando el error real, sin amplificar.")
        print("  - Vectores (V) con la limpia oculta y error amplificado: dirección y magnitud del ruido por")
        print("    punto; con ruido laplaciano destacan los pocos desplazamientos grandes de la cola pesada.")
        print("  - Alterna 2 y 3 sin mover la cámara para ver qué corrige MLS y qué no (los outliers persisten).")
        print("  - Lado a lado (M) + color por elevación (C) para comparar la forma general del relieve.")

    def run(self):
        self.print_help()
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name=self.window_name, width=1400, height=900)
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([0.05, 0.05, 0.05])
        opt.point_size = 2.0

        for key, callback in {
            "1": self._toggle("clean"),
            "2": self._toggle("degraded"),
            "3": self._toggle("denoised"),
            "V": self._toggle("lines"),
            "C": self._toggle_color_mode,
            "M": self._toggle_layout,
            "E": self._scale_error(2.0),
            "D": self._scale_error(0.5),
        }.items():
            self.vis.register_key_callback(ord(key), callback)

        self.refresh(reset_view=True)
        self.vis.run()
        self.vis.destroy_window()

def view_clean_vs_degraded(clean, degraded, is_outlier, denoised=None, window_name="Limpia vs. degradada",
                           max_points=100_000, max_lines=3_000, error_amplify=1.0, seed=42):
    CleanVsDegradedViewer(
        clean, degraded, is_outlier, denoised, window_name, max_points, max_lines, error_amplify, seed
    ).run()
