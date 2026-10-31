from pathlib import Path

# Configuración de rutas y dataset

#TERRAIN = "Northern_California_Coast_40M"
#TERRAIN = "San_Isabel_NF_CO_Debris_100M"
TERRAIN = "Sunset_Crater_Volcano_National_Monument_1M"
BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "analytics" / TERRAIN
FILE_PATH = DATA_DIR / TERRAIN / "points.laz"

# Muestreo y visualización
VIS_POINTS = 1_200_000
NN_SAMPLE = 125_000
SEED = 42

# Parámetros del parche y ruido gaussiano
PATCH_SIZE = 100.0        # Lado del patch en metros
MIN_POINTS = 1000         # Mínimo de puntos para aceptar un patch
NOISE_SIGMA = 0.1       # Desviación estándar del ruido (metros)
NOISE_AXES = "xyz"       # Opciones: "xyz" o "z"
PATCH_VIS_POINTS = 30_000
PATCH_SEED = 42
MAX_ATTEMPTS = 5
MAX_DISPLAY = 5_000      # Puntos para gráficos de diagnóstico
AMPLIFY = 8              # Amplificación visual del desplazamiento
NOISE_SEED = 42

NOISE_LAPLACE_SCALE = 0.15  # Parámetro 'b' para la distribución Laplaciana
OUTLIER_RATIO = 0.05        # Fracción de puntos espurios a inyectar (por cada tipo de outlier, sobre el patch limpio)

# --- Control de Outliers Topográficos ---
OUTLIER_Z_ABOVE = 3.0           # Máximo metros sobre la cota local para outliers globales
OUTLIER_Z_BELOW = 1.5           # Máximo metros bajo la cota local (multipath)
OUTLIER_CLUSTERS_COUNT = 3      # Cantidad de clústeres independientes para outliers agrupados

# Combinaciones de degradación a experimentar: cada lista es un experimento independiente.
# Componentes disponibles: "gaussian" (NOISE_SIGMA), "laplacian" (NOISE_LAPLACE_SCALE),
# "global_outliers" y "clustered_outliers" (OUTLIER_RATIO). Los ruidos posicionales se suman.
NOISE_COMBINATIONS = [
    ["gaussian"],
    ["laplacian"],
    ["gaussian", "laplacian"],
    ["gaussian", "clustered_outliers"],
    ["gaussian", "laplacian", "clustered_outliers"],
    ["gaussian", "laplacian", "global_outliers", "clustered_outliers"],
]

# -------------------------------------------------------------------
# Visualización de resultados
SHOW_FIGURES = True          # Mostrar las figuras de Matplotlib en pantalla (siempre se guardan en OUTPUT_DIR)
OPEN3D_VIEWER = True         # Abrir el visor 3D interactivo (limpia vs. degradada vs. MLS) en cada experimento
VIEW_MAX_POINTS = 100_000    # Puntos del patch mostrados en el visor 3D
VIEW_MAX_LINES = 3_000       # Vectores de desplazamiento (limpio -> degradado) mostrados en el visor 3D
VIEW_ERROR_AMPLIFY = 1.0     # Amplificación inicial del desplazamiento limpio -> degradado en el visor 3D (ajustable con E/D)

# -------------------------------------------------------------------
# Para MLS
MLS_K_NEIGHBORS = 40     # Número de vecinos a buscar (k)
MLS_H_PARAM = None       # Ancho de banda gaussiano fijo (m); None = automático según el espaciado
MLS_H_FACTOR = 5.0       # h automático = MLS_H_FACTOR × espaciado mediano entre vecinos de la nube de entrada

# -------------------------------------------------------------------
# Métricas
DCD_ALPHA = 10.0         # Sensibilidad a la distancia de la DCD (1/m)
NORMAL_K_NEIGHBORS = 30  # Vecinos para estimar normales en UNAE

# Inicialización de directorios
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)