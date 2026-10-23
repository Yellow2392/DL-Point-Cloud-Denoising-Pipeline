import laspy
import numpy as np

def load_las_data(file_path):
    las = laspy.read(file_path)
    points = np.column_stack((las.x, las.y, las.z)).astype(np.float64)
    intensity = np.asarray(las.intensity)
    classification = np.asarray(las.classification)
    
    return points, intensity, classification, las.header

def print_metadata(header, n_points, file_path):
    print("=" * 70)
    print(f"CARGANDO DATASET: {file_path}")
    print(f"Número de puntos: {n_points:,}")
    print("=" * 70)
    print(f"Versión LAS: {header.major_version}.{header.minor_version}")
    print(f"Formato de punto: {header.point_format.id}")
    print(f"Point count header: {header.point_count:,}")
    print(f"Escala: {header.scales}")
    print(f"Offset: {header.offsets}")
    print(f"Min: {header.mins}")
    print(f"Max: {header.maxs}")