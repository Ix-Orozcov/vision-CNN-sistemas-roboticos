# CNN de navegación en Python: dataset, entrenamiento, robustez y cuantización

Scripts para:
- Construir en **CoppeliaSim** la base de datos de señales de navegación.
- Entrenar con **Keras** la CNN que las clasifica.
- Medir su desempeño y su robustez ante perturbaciones.
- Cuantizarla a **INT8** con TensorFlow Lite.

El modelo INT8 resultante es el que después se exporta al microcontrolador S32K344 (ver el README de
`navegacion_s32k344`).

| Clase | ID | Señal | Acción del robot |
|---|---|---|---|
| `avanzar` | 0 | Flecha hacia arriba | Seguir avanzando |
| `izquierda` | 1 | Flecha a la izquierda | Girar 90° a la izquierda |
| `derecha` | 2 | Flecha a la derecha | Girar 90° a la derecha |
| `obstaculo` | 3 | Caja frente al robot | Detenerse |

---

## 1. Archivos

| Script | Qué hace |
|---|---|
| `capturar_dataset.py` | Captura en CoppeliaSim el dataset completo (train / validation / test), ya redimensionado a 64×64 y con ruido en el 20 % de las imágenes. |
| `capturar_perturbado.py` | Captura un conjunto **nuevo** de imágenes perturbadas (iluminación, distancia, ángulo de cámara, posición del patrón) para medir la robustez. |
| `agregar_ruido.py` | Agrega ruido gaussiano y sal y pimienta a las imágenes de una carpeta (guarda copias; no toca los originales). |
| `cnn_navegacion.py` | Entrena la CNN, la evalúa en test y en el conjunto perturbado, y genera todas las métricas y gráficas. |
| `cuantizar_cnn.py` | Convierte el modelo a TFLite FP32 e INT8 y compara tamaño, RAM, Flash, tiempo y accuracy. |
| `f1_int8_tflite.py` | Calcula el accuracy y el F1 del modelo INT8 de TFLite en el conjunto de prueba. |

---

## 2. Requisitos

```
pip install tensorflow scikit-learn matplotlib pillow numpy coppeliasim-zmqremoteapi-client
```
- Python 3.10 o más reciente.
- **CoppeliaSim Edu 4.10** para los scripts de captura. Debe estar abierto en una **escena nueva**,
  con la simulación detenida; el servidor ZMQ (puerto 23000) arranca solo con CoppeliaSim.

> **Rutas en Windows:** usa cadenas *raw* (`r"C:\Users\..."`) o diagonales normales
> (`"C:/Users/..."`). Sin la `r`, `\U` se interpreta como un carácter especial y Python marca error.

---

## 3. Flujo de trabajo

```
capturar_dataset.py  ->  dataset_run/ (train, validation, test)
capturar_perturbado.py -> dataset_perturbado/
        |                         |
        +-----> cnn_navegacion.py <+   ->  resultados_cnn/modelo_cnn.keras + métricas
                       |
                cuantizar_cnn.py       ->  modelo_fp32.tflite, modelo_int8.tflite, comparacion_fp32_int8.csv
                       |
                f1_int8_tflite.py      ->  F1 del modelo INT8
                       |
     navegacion_s32k344/pc/exportar_modelo_c.py  ->  modelo en C para el micro
```

### 3.1 Capturar el dataset

```
python capturar_dataset.py --output dataset_run
```

| Opción | Por defecto | Significado |
|---|---|---|
| `--frames-per-group` | 25 | Fotos por escenario y clase |
| `--resolution` | 320 | Resolución de la cámara virtual |
| `--save-size` | 64 | Tamaño final de las imágenes (Lanczos) |
| `--noise-fraction` | 0.20 | Fracción de imágenes con ruido |
| `--seed` | 73 | Semilla (mismo valor = mismo dataset) |
| `--build-only` | -- | Solo construye la escena, sin capturar |

**Cómo se construye:**
- Se capturan **20 escenarios por clase**, de 25 fotos cada uno: **2000 imágenes**, 500 por clase.
- En cada escenario cambian los tonos del piso y del fondo.
- En cada foto cambian:
  - la distancia a la señal (1.15–1.85 m);
  - la posición lateral (±0.12 m);
  - el ángulo de la cámara (giro ±10°, inclinación ±9°, rotación ±14°);
  - la altura de la cámara (0.32–0.62 m).
- El 20 % de las imágenes lleva ruido gaussiano (σ = 0.05) y sal y pimienta (2 % de los píxeles).
- Los escenarios completos se asignan a una sola partición, para que imágenes casi iguales no
  queden en train y test a la vez:
  - **14 a train** (1400 imágenes);
  - **3 a validation** (300);
  - **3 a test** (300).

**Salida:**
```
dataset_run/
    train/       avanzar/ izquierda/ derecha/ obstaculo/
    validation/  avanzar/ izquierda/ derecha/ obstaculo/
    test/        avanzar/ izquierda/ derecha/ obstaculo/
    manifest.csv       parámetros de cada imagen (distancia, ángulos, ruido...)
    vista_previa.png
    config.json
    escena_dataset.ttt
```

### 3.2 Capturar el conjunto perturbado

```
python capturar_perturbado.py --output dataset_perturbado --total 100 --intensidad moderada
```
- Cada imagen aplica **una** perturbación, y las demás condiciones quedan en su rango nominal:
  - `iluminacion`: luces tenues o intensas, con tono cálido o frío.
  - `distancia`: señal mucho más cerca o más lejos que en el entrenamiento.
  - `angulo_camara`: giro, inclinación, rotación y altura de la cámara.
  - `posicion_patron`: señal desplazada y girada.
- Las clases y las perturbaciones se reparten por igual. Con `--total 100` salen 25 imágenes por
  clase y 25 por perturbación.
- `--intensidad severa` usa rangos más extremos.
- Los archivos se llaman `000_iluminacion.png`, `004_distancia.png`, ... El nombre indica la
  perturbación, y así `cnn_navegacion.py` puede calcular el accuracy por tipo.

### 3.3 (Opcional) Agregar ruido a una carpeta

```
python agregar_ruido.py
```
Pide la ruta de la carpeta y guarda las copias con ruido en otra carpeta. El tipo y la intensidad se
cambian al inicio del script (`TIPO_RUIDO`, `SIGMA`, `PROB_SP`). `capturar_dataset.py` ya aplica este
mismo ruido, así que solo hace falta para otras imágenes.

### 3.4 Entrenar y evaluar la CNN

Primero ajusta las rutas al inicio de `cnn_navegacion.py`:
```python
RUTA_DATASET    = Path(r"...\dataset_run")          # carpeta con train/validation/test
RUTA_PERTURBADO = Path(r"...\dataset_perturbado")
EPOCAS = 100      # en las pruebas se usaron 100
BATCH  = 32
```
Luego:
```
python cnn_navegacion.py
```

**Arquitectura** (51 508 parámetros):

| Capa | Salida | Parámetros |
|---|---|---|
| Entrada en gris, normalizada (/255) | 64×64×1 | -- |
| Conv2D 8 filtros 3×3 + ReLU | 62×62×8 | 80 |
| MaxPooling 2×2 | 31×31×8 | -- |
| Conv2D 16 filtros 3×3 + ReLU | 29×29×16 | 1 168 |
| MaxPooling 2×2 | 14×14×16 | -- |
| Flatten | 3136 | -- |
| Dense 16 + ReLU | 16 | 50 192 |
| Dense 4 + Softmax | 4 | 68 |

**Entrenamiento:**
- Optimizador Adam, entropía cruzada como pérdida y lotes de 32.
- Aumento de datos: brillo, contraste, zoom, desplazamiento y rotación pequeña. **Sin volteo
  horizontal**, porque convertiría una flecha izquierda en derecha.
- Si usas `EarlyStopping`, define la variable `parada` **antes** de llamar a `modelo.fit(...)` y
  pásala en `callbacks=[parada]`.

**Resultados en `resultados_cnn/`:**

| Archivo | Contenido |
|---|---|
| `modelo_cnn.keras` | Modelo entrenado |
| `curvas_aprendizaje.png` | Accuracy y loss por época (train y validation) |
| `matriz_confusion.png` | Matriz de confusión en test |
| `accuracy_por_clase.png`, `precision.png`, `recall.png`, `f1_score.png` | Métricas por clase en test |
| `inferencia_ejemplos.png` | Ejemplos de predicciones |
| `robustez_perturbaciones.png` | Accuracy nominal, perturbado y por tipo de perturbación |
| `matriz_confusion_perturbado.png` | Matriz de confusión del conjunto perturbado |
| `accuracy_clase_nominal_vs_perturbado.png` | Comparación por clase |
| `inferencia_perturbado.png` | Ejemplos perturbados |

La consola también imprime el reporte de clasificación (precision, recall y F1 por clase), el tiempo
de entrenamiento y la robustez:

**Δ = Accuracy_nominal − Accuracy_perturbado**

### 3.5 Cuantizar a INT8

Ajusta `RUTA_DATASET` y `RUTA_PERTURBADO` en `cuantizar_cnn.py` igual que antes y ejecuta:
```
python cuantizar_cnn.py
```
- Convierte `modelo_cnn.keras` a TFLite **FP32** y a **INT8** con cuantización entera completa.
  La entrada y la salida quedan en int8 y la calibración usa 300 imágenes de train.
- Mide para cada versión el tamaño de los pesos, la RAM de activaciones estimada, la Flash (archivo
  `.tflite`), el tiempo de inferencia (1 hilo) y el accuracy en test y en el conjunto perturbado.
- Genera `modelo_fp32.tflite`, `modelo_int8.tflite` y `comparacion_fp32_int8.csv`.

### 3.6 F1 del modelo INT8

Ajusta `RUTA_TEST` y `RUTA_MODELO` al inicio del archivo y ejecuta:
```
python f1_int8_tflite.py
```
Imprime el accuracy, el F1 macro y el reporte por clase del modelo INT8, con el mismo
preprocesamiento que el entrenamiento.

### 3.7 Pasar el modelo al microcontrolador

```
python navegacion_s32k344/pc/exportar_modelo_c.py --tflite "<...>\resultados_cnn\modelo_int8.tflite" --test "<...>\dataset_run\test"
```

---

## 4. Resultados obtenidos

| Métrica | Valor |
|---|---|
| Accuracy en test (300 imágenes), FP32 | 98.33 % (F1 macro 98.33 %) |
| Accuracy en test, INT8 | 98.00 % |
| Accuracy en el conjunto perturbado (100 imágenes) | 95.00 % (FP32 e INT8) |
| Robustez Δ | 0.033 |
| Accuracy por perturbación | iluminación 100 %, ángulo de cámara 100 %, posición 96 %, distancia 84 % |
| Tamaño de los pesos FP32 → INT8 | 201.2 KB → 50.3 KB (4.0× menor) |
| Archivo `.tflite` FP32 → INT8 | 205.6 KB → 56.3 KB |
| RAM de activaciones (estimada) | 150.2 KB → 37.5 KB |
| Tiempo de inferencia en la PC | 0.117 ms → 0.104 ms |

---

## 5. Problemas comunes

| Error | Solución |
|---|---|
| `SyntaxError: (unicode error) 'unicodeescape'` | Ruta de Windows sin `r` al inicio: usa `r"C:\..."`. |
| `FileNotFoundError` al crear carpetas | Revisa que la ruta exista y esté bien escrita (por ejemplo `Fisica` contra `Física`); las carpetas se crean con `mkdir(parents=True, exist_ok=True)`. |
| `NameError: name 'parada' is not defined` | Define el `EarlyStopping` antes de `modelo.fit`. |
| Los imports de `tensorflow.keras` salen en amarillo en el editor | Es solo el analizador del editor; el código funciona. |
| Los scripts de captura no conectan | CoppeliaSim debe estar abierto (servidor ZMQ en el puerto 23000) y con la simulación detenida. |
| Buen accuracy en test pero bajo en el perturbado | Aumenta la variación del dataset (ángulos, distancia, ruido) y usa aumento de datos. La distancia es la perturbación más difícil. |
