import laspy
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# CONFIGURACIÓN
# ============================================================

#terrain = "Northern_California_Coast_40M"
#terrain = "San_Isabel_NF_CO_Debris_100M"
terrain = "Sunset_Crater_Volcano_National_Monument_80M"
file_path = Path(f"data/{terrain}/points.laz")
output_dir = Path(f"analytics/{terrain}")
output_dir.mkdir(parents=True, exist_ok=True)

# Máximo de puntos usados para visualización 3D
VIS_POINTS = 1_200_000

# Puntos usados para estadísticas de vecino más cercano
NN_SAMPLE = 125_000

# Semilla para que el muestreo sea reproducible
SEED = 42
rng = np.random.default_rng(SEED)

# ============================================================
# LECTURA
# ============================================================

print("=" * 70)
print("CARGANDO DATASET")
print("=" * 70)

las = laspy.read(file_path)

points = np.column_stack((las.x, las.y, las.z)).astype(np.float64)

intensity = np.asarray(las.intensity)
classification = np.asarray(las.classification)

n = len(points)

print(f"Archivo: {file_path}")
print(f"Número de puntos: {n:,}")

# ============================================================
# METADATOS LAS
# ============================================================

print("\n" + "=" * 70)
print("METADATOS LAS")
print("=" * 70)

print(f"Versión LAS: {las.header.major_version}.{las.header.minor_version}")
print(f"Formato de punto: {las.header.point_format.id}")
print(f"Point count header: {las.header.point_count:,}")
print(f"Escala: {las.header.scales}")
print(f"Offset: {las.header.offsets}")
print(f"Min: {las.header.mins}")
print(f"Max: {las.header.maxs}")

# ============================================================
# EXTENSIÓN ESPACIAL
# ============================================================

xyz_min = points.min(axis=0)
xyz_max = points.max(axis=0)
extent = xyz_max - xyz_min

print("\n" + "=" * 70)
print("EXTENSIÓN ESPACIAL")
print("=" * 70)

print(f"X: {xyz_min[0]:.3f} -> {xyz_max[0]:.3f}  ({extent[0]:.3f} m)")
print(f"Y: {xyz_min[1]:.3f} -> {xyz_max[1]:.3f}  ({extent[1]:.3f} m)")
print(f"Z: {xyz_min[2]:.3f} -> {xyz_max[2]:.3f}  ({extent[2]:.3f} m)")

area_km2 = (extent[0] * extent[1]) / 1e6
print(f"Área aproximada XY: {area_km2:.3f} km²")

# ============================================================
# ESTADÍSTICAS BÁSICAS
# ============================================================

def print_stats(name, values):
    values = np.asarray(values)
    print(f"\n{name}")
    print(f"  min    : {np.min(values):.4f}")
    print(f"  p01    : {np.percentile(values, 1):.4f}")
    print(f"  p05    : {np.percentile(values, 5):.4f}")
    print(f"  median : {np.median(values):.4f}")
    print(f"  mean   : {np.mean(values):.4f}")
    print(f"  p95    : {np.percentile(values, 95):.4f}")
    print(f"  p99    : {np.percentile(values, 99):.4f}")
    print(f"  max    : {np.max(values):.4f}")
    print(f"  std    : {np.std(values):.4f}")

print("\n" + "=" * 70)
print("ESTADÍSTICAS")
print("=" * 70)

print_stats("Elevación Z (m)", points[:, 2])
print_stats("Intensidad", intensity)

# ============================================================
# CLASIFICACIONES
# ============================================================

print("\n" + "=" * 70)
print("CLASIFICACIONES")
print("=" * 70)

classes, counts = np.unique(classification, return_counts=True)

for c, count in zip(classes, counts):
    percentage = 100 * count / n
    print(f"Clase {int(c):2d}: {count:>12,} puntos ({percentage:6.2f}%)")

# ============================================================
# DENSIDAD PROMEDIO
# ============================================================

print("\n" + "=" * 70)
print("DENSIDAD APROXIMADA")
print("=" * 70)

if area_km2 > 0:
    density = n / (area_km2 * 1_000_000)
    print(f"Densidad promedio XY: {density:.3f} puntos/m²")

# ============================================================
# MUESTREO PARA ANÁLISIS LOCAL
# ============================================================

sample_size = min(NN_SAMPLE, n)
sample_indices = rng.choice(n, sample_size, replace=False)
sample_points = points[sample_indices]

print("\n" + "=" * 70)
print("VECINO MÁS CERCANO")
print("=" * 70)
print(f"Muestra: {sample_size:,} puntos")

# Construimos el KD-tree SOLO sobre la muestra.
# Esto es una estimación del spacing local y no una medición exacta
# del vecino más cercano en los 40M puntos.
sample_pcd = o3d.geometry.PointCloud()
sample_pcd.points = o3d.utility.Vector3dVector(sample_points)

tree = o3d.geometry.KDTreeFlann(sample_pcd)

nn_distances = []

for i in range(sample_size):
    k, idx, dist2 = tree.search_knn_vector_3d(sample_points[i], 2)

    if k > 1:
        # idx[0] es el propio punto; idx[1] es el vecino de la muestra
        nn_distances.append(np.sqrt(dist2[1]))

nn_distances = np.asarray(nn_distances)

print_stats("Distancia al vecino más cercano en la muestra (m)",
            nn_distances)

# ============================================================
# VISUALIZACIÓN 3D
# ============================================================

print("\n" + "=" * 70)
print("PREPARANDO VISUALIZACIÓN")
print("=" * 70)

vis_size = min(VIS_POINTS, n)
vis_indices = rng.choice(n, vis_size, replace=False)

points_vis = points[vis_indices]
z_vis = points_vis[:, 2]

# Percentiles para evitar que valores extremos dominen el mapa
z_low = np.percentile(z_vis, 2)
z_high = np.percentile(z_vis, 98)

z_norm = np.clip((z_vis - z_low) / (z_high - z_low + 1e-12), 0, 1)

# Heat map de elevación
colors = plt.get_cmap("turbo")(z_norm)[:, :3]

pcd_vis = o3d.geometry.PointCloud()
pcd_vis.points = o3d.utility.Vector3dVector(points_vis)
pcd_vis.colors = o3d.utility.Vector3dVector(colors)

print(f"Puntos visualizados: {vis_size:,}")
print(f"Rango de color Z: P02={z_low:.3f} m, P98={z_high:.3f} m")

print("\nControles:")
print("  - Arrastrar: rotar")
print("  - Rueda: zoom")
print("  - Shift + arrastrar: trasladar")
print("  - Q / ESC: cerrar")

vis = o3d.visualization.Visualizer()
vis.create_window(
    window_name=f"{terrain} - Elevación (Heat Map)",
    width=1400,
    height=900,
)
vis.add_geometry(pcd_vis)

# Cambiar fondo a negro: RGB [0, 0, 0]
opt = vis.get_render_option()
opt.background_color = np.asarray([0, 0, 0])

vis.run()
vis.destroy_window()

# ============================================================
# GRÁFICOS ESTADÍSTICOS
# ============================================================
print("\nGenerando gráficos...")

fig, axs = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle(
    f"Perfil estadístico - {terrain}\n"
    f"{n:,} puntos",
    fontsize=15
)

# Elevación
axs[0, 0].hist(points[:, 2], bins=150)
axs[0, 0].set_title("Distribución de elevación (Z)")
axs[0, 0].set_xlabel("Z (m)")
axs[0, 0].set_ylabel("Número de puntos")

# Intensidad
axs[0, 1].hist(intensity, bins=150)
axs[0, 1].set_title("Distribución de intensidad")
axs[0, 1].set_xlabel("Intensidad")
axs[0, 1].set_ylabel("Número de puntos")

# Clasificaciones
axs[1, 0].bar(classes.astype(str), counts)
axs[1, 0].set_title("Distribución de clasificación")
axs[1, 0].set_xlabel("Clase LAS")
axs[1, 0].set_ylabel("Número de puntos")

# Vecino más cercano
axs[1, 1].hist(nn_distances, bins=100)
axs[1, 1].set_title(
    f"Distancia al vecino más cercano\n"
    f"(muestra de {sample_size:,})"
)
axs[1, 1].set_xlabel("Distancia (m)")
axs[1, 1].set_ylabel("Número de puntos")

plt.tight_layout()

stats_plot = output_dir / "profile_statistics.png"
plt.savefig(stats_plot, dpi=150, bbox_inches="tight")
plt.show()

print(f"\nGráfico guardado en: {stats_plot}")
print("Perfilado terminado.")

# ============================================================
# PATCH LOCAL + RUIDO GAUSSIANO
# Insertar después de cargar points y crear output_dir
# ============================================================

# --- Configuración experimental ---
PATCH_SIZE = 50.0       # Lado del patch en metros
MIN_POINTS = 600        # Mínimo de puntos para aceptar un patch
NOISE_SIGMA = 0.05      # Desviación estándar del ruido (metros)
NOISE_AXES = "xyz"      # "xyz" o "z"
PATCH_VIS_POINTS = 30_000
PATCH_SEED = 42
MAX_ATTEMPTS = 5
MAX_DISPLAY = 5_000      # puntos para visualización
AMPLIFY = 8              # solo para el gráfico de desplazamientos
NOISE_SEED = 42

# ------------------------------------------------------------
# 1. Partición espacial en patches cuadrados sobre XY
# ------------------------------------------------------------

patch_rng = np.random.default_rng(PATCH_SEED)

n = len(points)
half = PATCH_SIZE / 2

patch_clean = None
patch_indices = None

# Candidatos aleatorios sin construir una cuadrícula global.
candidate_indices = patch_rng.choice(
    n,
    size=min(MAX_ATTEMPTS, n),
    replace=False
)

for candidate_idx in candidate_indices:
    center_x, center_y = points[candidate_idx, :2]

    # Máscara de un cuadrado de PATCH_SIZE x PATCH_SIZE.
    mask = (
        (np.abs(points[:, 0] - center_x) <= half) &
        (np.abs(points[:, 1] - center_y) <= half)
    )

    indices = np.flatnonzero(mask)

    if len(indices) >= MIN_POINTS:
        patch_indices = indices
        patch_clean = points[indices].copy()
        patch_center = (center_x, center_y)
        break

if patch_clean is None:
    raise ValueError(
        f"No se encontró un patch de {PATCH_SIZE} m con "
        f"al menos {MIN_POINTS} puntos tras {MAX_ATTEMPTS} intentos. "
        "Prueba un patch mayor o reduce MIN_POINTS."
    )

print(f"Patch extraído: {len(patch_clean):,} puntos")
print(f"Centro XY: {patch_center}")
print(f"Extensión XY: {np.ptp(patch_clean[:, 0]):.3f} x "
      f"{np.ptp(patch_clean[:, 1]):.3f} m")

noise_rng = np.random.default_rng(NOISE_SEED)

# Conservar la referencia limpia
patch_noisy = patch_clean.copy()

if NOISE_AXES == "xyz":
    noise = noise_rng.normal(
        0.0, NOISE_SIGMA, size=patch_clean.shape
    )
    patch_noisy += noise
elif NOISE_AXES == "z":
    noise = noise_rng.normal(
        0.0, NOISE_SIGMA, size=len(patch_clean)
    )
    patch_noisy[:, 2] += noise
else:
    raise ValueError("NOISE_AXES debe ser 'xyz' o 'z'.")

# Desplazamientos reales, calculados antes de cualquier amplificación
delta = patch_noisy - patch_clean
delta_norm = np.linalg.norm(delta, axis=1)

print(f"Sigma configurado: {NOISE_SIGMA:.4f} m")
print(f"Desviación estándar observada X: {delta[:, 0].std():.4f} m")
print(f"Desviación estándar observada Y: {delta[:, 1].std():.4f} m")
print(f"Desviación estándar observada Z: {delta[:, 2].std():.4f} m")
print(f"Desplazamiento 3D medio: {delta_norm.mean():.4f} m")
print(f"Percentil 95 del desplazamiento: "
      f"{np.percentile(delta_norm, 95):.4f} m")

# Submuestreo solo para los gráficos
rng_vis = np.random.default_rng(123)
n_vis = min(MAX_DISPLAY, len(patch_clean))
idx_vis = rng_vis.choice(len(patch_clean), n_vis, replace=False)

clean_vis = patch_clean[idx_vis]
noisy_vis = patch_noisy[idx_vis]
delta_vis = delta[idx_vis]

# Misma escala de color por elevación en ambos paneles
z_min = min(clean_vis[:, 2].min(), noisy_vis[:, 2].min())
z_max = max(clean_vis[:, 2].max(), noisy_vis[:, 2].max())

fig, axs = plt.subplots(1, 3, figsize=(18, 5))

sc1 = axs[0].scatter(
    clean_vis[:, 0], clean_vis[:, 1],
    c=clean_vis[:, 2], s=2,
    cmap="turbo", vmin=z_min, vmax=z_max,
    rasterized=True
)
axs[0].set_title("Patch limpio")
axs[0].set_xlabel("X (m)")
axs[0].set_ylabel("Y (m)")
axs[0].set_aspect("equal", adjustable="box")

axs[1].scatter(
    noisy_vis[:, 0], noisy_vis[:, 1],
    c=noisy_vis[:, 2], s=2,
    cmap="turbo", vmin=z_min, vmax=z_max,
    rasterized=True
)
axs[1].set_title(f"Patch con ruido gaussiano ({NOISE_AXES})")
axs[1].set_xlabel("X (m)")
axs[1].set_ylabel("Y (m)")
axs[1].set_aspect("equal", adjustable="box")

fig.colorbar(sc1, ax=axs[:2], label="Elevación Z (m)", shrink=0.8)

# Desplazamientos: líneas entre las posiciones limpias y ruidosas.
# Se amplifican únicamente para facilitar su inspección visual.
if NOISE_AXES == "xyz":
    for p, d in zip(clean_vis, delta_vis):
        axs[2].plot(
            [p[0], p[0] + AMPLIFY * d[0]],
            [p[1], p[1] + AMPLIFY * d[1]],
            linewidth=0.5, alpha=0.5
        )
    axs[2].scatter(
        clean_vis[:, 0], clean_vis[:, 1],
        s=3, label="Posición limpia"
    )
    axs[2].scatter(
        clean_vis[:, 0] + AMPLIFY * delta_vis[:, 0],
        clean_vis[:, 1] + AMPLIFY * delta_vis[:, 1],
        s=3, label=f"Desplazamiento XY x{AMPLIFY}"
    )
    axs[2].set_title("Desplazamiento horizontal amplificado")
    axs[2].set_xlabel("X (m)")
    axs[2].set_ylabel("Y (m)")
else:
    axs[2].hist(delta[:, 2], bins=60)
    axs[2].axvline(0, linestyle="--")
    axs[2].set_title("Distribución del ruido vertical")
    axs[2].set_xlabel("Desplazamiento Z (m)")
    axs[2].set_ylabel("Número de puntos")

if NOISE_AXES == "xyz":
    axs[2].set_aspect("equal", adjustable="datalim")
    axs[2].legend(markerscale=2, fontsize=8)

fig.suptitle(
    f"Patch de {len(patch_clean):,} puntos | "
    f"sigma={NOISE_SIGMA:.3f} m | semilla={NOISE_SEED}"
)
fig.tight_layout()

plot_path = output_dir / "patch_gaussian_diagnostic.png"
fig.savefig(plot_path, dpi=160, bbox_inches="tight")
plt.show()
plt.close(fig)

# Guardar los datos completos, no el submuestreo de visualización
np.savez_compressed(
    output_dir / "patch_gaussian_data.npz",
    clean=patch_clean,
    noisy=patch_noisy,
    source_indices=patch_indices,
    sigma=np.array(NOISE_SIGMA),
    axes=np.array(NOISE_AXES),
    patch_size=np.array(PATCH_SIZE)
)

print(f"Figura: {plot_path}")
print(f"Datos: {output_dir / 'patch_gaussian_data.npz'}")