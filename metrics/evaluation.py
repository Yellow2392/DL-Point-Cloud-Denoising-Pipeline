import csv
from metrics.rmse_z import rmse_z
from metrics.chamfer_distance import chamfer_distance, density_aware_chamfer_distance
from metrics.hausdorff_distance import hausdorff_distance
from metrics.unae import unae

def evaluate(pred, ref, dcd_alpha, normal_k):
    # Conjunto completo de métricas de una nube frente a la nube de referencia limpia
    return {
        "RMSE_z (m)": rmse_z(pred, ref),
        "CD (m)": chamfer_distance(pred, ref),
        "DCD": density_aware_chamfer_distance(pred, ref, dcd_alpha),
        "HD (m)": hausdorff_distance(pred, ref),
        "HD95 (m)": hausdorff_distance(pred, ref, percentile=95.0),
        "UNAE (°)": unae(pred, ref, normal_k),
    }

def print_metrics_table(results):
    # results: {nombre_de_fila: {métrica: valor}}
    names = list(results)
    columns = list(results[names[0]])
    name_width = max(len(name) for name in names)

    print(f"{'':<{name_width}}  " + "  ".join(f"{col:>11}" for col in columns))
    for name in names:
        print(f"{name:<{name_width}}  " + "  ".join(f"{results[name][col]:>11.4f}" for col in columns))

def save_metrics_csv(results, output_path):
    columns = list(next(iter(results.values())))
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["nube"] + columns)
        for name, values in results.items():
            writer.writerow([name] + [f"{values[col]:.6f}" for col in columns])
