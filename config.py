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
OUTLIER_RATIO = 0.05           # Porcentaje de puntos espurios a inyectar
OUTLIER_TYPE = "clustered"     # "global" o "clustered"

# -------------------------------------------------------------------
# Para MLS
MLS_K_NEIGHBORS = 40     # Número de vecinos a buscar (k)
MLS_H_PARAM = 0.08       # Ancho de banda de la función de peso gaussiana (m)

# Inicialización de directorios
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)