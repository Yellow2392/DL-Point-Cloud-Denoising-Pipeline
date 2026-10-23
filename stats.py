import numpy as np
import open3d as o3d

def calculate_spatial_extent(points):
    xyz_min = points.min(axis=0)
    xyz_max = points.max(axis=0)
    extent = xyz_max - xyz_min
    area_km2 = (extent[0] * extent[1]) / 1e6
    
    print("\n" + "=" * 70)
    print("EXTENSIÓN ESPACIAL")
    print("=" * 70)
    print(f"X: {xyz_min[0]:.3f} -> {xyz_max[0]:.3f}  ({extent[0]:.3f} m)")
    print(f"Y: {xyz_min[1]:.3f} -> {xyz_max[1]:.3f}  ({extent[1]:.3f} m)")
    print(f"Z: {xyz_min[2]:.3f} -> {xyz_max[2]:.3f}  ({extent[2]:.3f} m)")
    print(f"Área aproximada XY: {area_km2:.3f} km²")
    
    return extent, area_km2

def calculate_density(n_points, area_km2):
    if area_km2 > 0:
        density = n_points / (area_km2 * 1_000_000)
        print(f"Densidad promedio XY: {density:.3f} puntos/m²")
        return density
    return 0.0

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

def calculate_nearest_neighbor(points, sample_size, seed):
    rng = np.random.default_rng(seed)
    n = len(points)
    actual_sample_size = min(sample_size, n)
    sample_indices = rng.choice(n, actual_sample_size, replace=False)
    sample_points = points[sample_indices]
    
    sample_pcd = o3d.geometry.PointCloud()
    sample_pcd.points = o3d.utility.Vector3dVector(sample_points)
    tree = o3d.geometry.KDTreeFlann(sample_pcd)
    
    nn_distances = []
    for i in range(actual_sample_size):
        k, idx, dist2 = tree.search_knn_vector_3d(sample_points[i], 2)
        if k > 1:
            nn_distances.append(np.sqrt(dist2[1]))
            
    return np.asarray(nn_distances), actual_sample_size