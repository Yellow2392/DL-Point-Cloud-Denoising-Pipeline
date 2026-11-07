import numpy as np

def outlier_detection_metrics(predicted_outlier, true_outlier):
    # Calidad de la fase de remoción frente a los outliers inyectados (verdad de terreno).
    # Precisión: fracción de lo eliminado que realmente era outlier.
    # Recall: fracción de los outliers inyectados que fue eliminada.
    # Inliers eliminados: fracción de puntos válidos del terreno perdidos por la remoción.
    tp = np.sum(predicted_outlier & true_outlier)
    fp = np.sum(predicted_outlier & ~true_outlier)
    fn = np.sum(~predicted_outlier & true_outlier)

    precision = tp / (tp + fp) if tp + fp > 0 else np.nan
    recall = tp / (tp + fn) if tp + fn > 0 else np.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else np.nan

    return {
        "Precisión": precision,
        "Recall": recall,
        "F1": f1,
        "Inliers elim. (%)": 100.0 * fp / max(np.sum(~true_outlier), 1),
    }
