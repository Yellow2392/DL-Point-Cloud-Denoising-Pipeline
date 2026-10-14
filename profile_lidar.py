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
