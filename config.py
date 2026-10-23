from pathlib import Path

# Configuración de rutas y dataset

#TERRAIN = "Northern_California_Coast_40M"
#TERRAIN = "San_Isabel_NF_CO_Debris_100M"
TERRAIN = "Sunset_Crater_Volcano_National_Monument_80M"
BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "analytics" / TERRAIN
FILE_PATH = DATA_DIR / TERRAIN / "points.laz"

# Muestreo y visualización
VIS_POINTS = 1_200_000
NN_SAMPLE = 125_000
SEED = 42

# Parámetros del parche y ruido gaussiano
PATCH_SIZE = 50.0        # Lado del patch en metros
MIN_POINTS = 600         # Mínimo de puntos para aceptar un patch
NOISE_SIGMA = 0.05       # Desviación estándar del ruido (metros)
NOISE_AXES = "xyz"       # Opciones: "xyz" o "z"
PATCH_VIS_POINTS = 30_000
PATCH_SEED = 42
MAX_ATTEMPTS = 5
MAX_DISPLAY = 5_000      # Puntos para gráficos de diagnóstico
AMPLIFY = 8              # Amplificación visual del desplazamiento
NOISE_SEED = 42

# Inicialización de directorios
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)