# Pipeline de Análisis, Degradación Sintética y Denoising en Nubes de Puntos 3D Topográficas

Implementación experimental y exploratoria desarrollada en el marco del trabajo de investigación para optar el grado de bachiller en Ciencia de la Computación en la Universidad de Ingeniería y Tecnología (UTEC):

> **Título:** *Pipeline robusto basado en Deep Learning para la reducción de ruido y preservación geométrica en nubes de puntos 3D topográficas*
> 
> 
> **Autor:** Jorge Sebastian Tenorio Romero
> 
> 
> **Asesor:** Dr. Henry Giovanny Gallegos Velgara
> 
> 
> **Año:** 2026
> 
> 

---

## 1. Contexto y Alcance de la Tesis

La captura de datos topográficos mediante sensores LiDAR aerotransportados (ALS) y terrestres (TLS) genera conjuntos masivos de coordenadas $(X, Y, Z)$ desorganizadas. Durante la adquisición, factores instrumentales, atmosféricos y oclusiones introducen degradaciones posicionales y mediciones anómalas que distorsionan la fidelidad geométrica de los modelos digitales de elevación (DEM/DTM).

Los métodos convencionales de filtrado y suavizado (como *Statistical Outlier Removal* o *Moving Least Squares*) suelen inducir **sobre-suavizado** (*over-smoothing*), desgastando rupturas de pendiente, valles y crestas escarpadas indispensables para el análisis geomorfológico.

Esta investigación desarrolla un pipeline computacional supervisado de aprendizaje profundo (*Deep Learning*) enfocado en:

1. Segmentación y normalización espacial de nubes masivas mediante parches locales (*local patches*).


2. Inyección y control estructurado de modelos estocásticos de ruido (Gaussiano, Laplaciano y valores atípicos / *outliers*).


3. Proyección mediante regresión de vectores de desplazamiento 3D (sin eliminar puntos válidos ni alterar la cardinalidad de la superficie) combinada con podas de *outliers*.


4. Optimización multiobjetivo con restricciones geométricas que salvaguarden cotas altimétricas ($RMSE_z$), consistencia de pendientes ($UNAE$) y dispersión volumétrica ($DCD$, $HD$).


Este repositorio implementa la fase experimental base: caracterización estadística de datasets LiDAR, simulación estocástica de degradaciones controladas y evaluación de algoritmos clásicos de referencia (*Moving Least Squares*).

---

## 2. Estructura del Proyecto

El código está modularizado para garantizar reproducibilidad y aislamiento de responsabilidades:

```text
.
├── data/                                 # Datasets crudos LiDAR (.laz / .las)
│   └── Sunset_Crater_Volcano_National_Monument_80M/
│       └── points.laz
├── analytics/                            # Salidas gráficas, diagnósticos y datos .npz
│   └── Sunset_Crater_Volcano_National_Monument_80M/
│       ├── profile_statistics.png
│       ├── patch_gaussian_diagnostic.png
│       ├── patch_gaussian_data.npz
│       └── denoising_mls_comparison.png
├── config.py                             # Hiperparámetros, rutas y semillas reproducibles
├── data_loader.py                        # Ingesta de archivos LAS/LAZ y metadatos
├── stats.py                              # Cálculos de extensión, densidad y KD-Trees
├── experiment.py                         # Extracción de parches e inyección de ruido
├── denoising.py                          # Algoritmos de restauración (MLS 1er orden)
├── visualization.py                      # Renders 3D Open3D y diagnósticos Matplotlib
├── main.py                               # Orquestador del pipeline
└── README.md

```

---

## 3. Módulos y Funcionalidades

### `config.py`

Centraliza la configuración del entorno para evitar valores fijos en el código:

* Rutas dinámicas de entrada/salida según el terreno evaluado (`TERRAIN`).
* Parámetros de visualización interactiva (`VIS_POINTS`) y muestreo de densidad (`NN_SAMPLE`).
* Semillas aleatorias (`SEED`, `PATCH_SEED`, `NOISE_SEED`) para garantizar reproducibilidad estocástica.
* Configuración de la partición espacial (`PATCH_SIZE`, `MIN_POINTS`).
* Hiperparámetros de ruido posicional (`NOISE_SIGMA`, `NOISE_LAPLACE_SCALE`, `NOISE_AXES`) y contaminación por *outliers* (`OUTLIER_RATIO`, `OUTLIER_TYPE`).
* Parámetros de suavizado para la línea base clásica (`MLS_K_NEIGHBORS`, `MLS_H_PARAM`).

### `data_loader.py`

* Lectura eficiente de nubes de puntos georreferenciadas en formato LAS/LAZ mediante `laspy`.
* Extracción y tipado de coordenadas $(X, Y, Z)$ en punto flotante (`float64`).
* Lectura de atributos complementarios (intensidad de haz, clases ASPRS).
* Inspección de metadatos de cabecera: versión de especificación LAS, escalas de cuantización, offsets y cajas delimitadoras.

### `stats.py`

* **Extensión espacial:** Cálculo de envolventes métricas mínimas/máximas y área proyectada en el plano horizontal ($km^2$).
* **Densidad:** Estimación promedio de muestreo espacial ($\text{puntos}/m^2$).
* **Distribuciones univariadas:** Cálculo de percentiles críticos ($P_{01}$, $P_{05}$, $P_{50}$, $P_{95}$, $P_{99}$), media y desviación estándar para elevación e intensidad.
* **Espaciado local:** Indexación espacial mediante `KDTreeFlann` sobre submuestras para calcular la distribución de distancias al vecino más cercano ($k=2$).

### `experiment.py`

Módulo encargado de generar los pares de entrenamiento y evaluación experimental:

* **Extracción de parches locales:** Búsqueda y recorte de ventanas espaciales delimitadas en $XY$ ($L \times L\text{ m}$) garantizando una cuota mínima de puntos (`min_points`).
* **Ruido Gaussiano:** Perturbación aditiva isótropa $p_i = x_i + e_i$ con $e_i \sim \mathcal{N}(0, \sigma^2 \mathbf{I})$, orientada a simular la incertidumbre y vibración instrumental del sensor.


* **Ruido Laplaciano:** Perturbación con colas pesadas $e_{i,k} \sim \text{Laplace}(0, b)$, modelando saltos abruptos en bordes o fallas geométricas.


* **Inyección de Outliers:** Generación de puntos espurios mediante un modelo de mezcla estocástica:


* *Globales:* Distribución uniforme $U(V)$ en el volumen espacial del terreno para simular polvo o rebotes atmosféricos.


* *Agrupados (Clustered):* Clústeres densos flotantes sobre el terreno para emular reflexiones en vegetación o micro-obstáculos.



### `models/moving_least_squares.py`

* Implementación de **Moving Least Squares (MLS) de 1er orden** como línea base analítica tradicional.


* Estructuración de vecindarios locales mediante KD-Tree.


* Estimación de planos tangentes y centroides ponderados por pesos gaussianos:

$$w_j = \exp\left(-\frac{\Vert{}\mathbf{p}_j - \mathbf{p}_i\Vert{}^2}{h^2}\right)$$


* Estimación de normales locales mediante descomposición espectral (PCA) sobre la matriz de covarianza ponderada y proyección ortogonal de los puntos degradados.

### `visualization.py`

* Renderizado interactivo 3D con mapas de color normalizados (vía percentiles para suprimir artefactos visuales) sobre fondo oscuro en `Open3D`.
* Diagnósticos estadísticos en `Matplotlib` (histogramas de elevación, intensidad, clases y espaciado de vecinos).
* Vistas comparativas multifiltro (original limpio, corrompido, restaurado por MLS) y gráficos vectoriales de desplazamiento angular y horizontal amplificado.

---

## 4. Requisitos e Instalación

Se recomienda el uso de un entorno virtual con Python 3.10 o superior:

```bash
# Crear y activar entorno virtual
python -m venv venv
venv\Scripts\activate       # En Windows
# source venv/bin/activate      # En Linux / macOS

# Instalar dependencias numéricas y geoespaciales
pip install numpy matplotlib open3d "laspy[lazrs]"

```

---

## 5. Ejecución

1. Ubicar la nube de puntos descargada (por ejemplo, desde OpenTopography) dentro del directorio correspondiente:
```text
data/<NOMBRE_DEL_TERRAIN>/points.laz

```
2. Configurar la variable `TERRAIN` y los hiperparámetros de simulación dentro de `config.py`.
3. Ejecutar el orquestador principal:
```bash
python main.py

```

El script imprimirá el perfil métrico en consola, abrirá la visualización 3D interactiva y almacenará las figuras y arrays resultantes en formato comprimido (`.npz`) dentro de `analytics/<NOMBRE_DEL_TERRAIN>/`.

# Quick Start
- Ejecutar para activar el entorno virtual sin usar Anaconda Prompt directamente en terminal: `(C:\Users\Soporte\miniconda3\shell\condabin\conda-hook.ps1) ; (conda activate lidar_env)`
- Usa `Ctrl + Shift + P` para elegir el intérprete de Python en VS Code. 