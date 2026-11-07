"""
Optimización para implementación embebida: cuantización FP32 -> INT8.

Usa el modelo ya entrenado por cnn_navegacion.py (resultados_cnn/modelo_cnn.keras),
lo convierte a TensorFlow Lite en dos versiones y las compara:

    - FP32: pesos y activaciones en punto flotante de 32 bits (4 bytes por valor)
    - INT8: cuantización entera completa (1 byte por valor), calibrada con imágenes de train

Mide para cada versión:
    Tamaño del modelo, RAM, Flash, Tiempo de inferencia y Accuracy

Requisitos:
    pip install tensorflow pillow numpy
"""

import time
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# ============================================================
# CONFIGURACIÓN (mismas rutas que en cnn_navegacion.py)
# ============================================================
RUTA_DATASET = Path(r"C:\Users\ixoro\Documents\UNI\7moSemestre\Inteligencia Artificial Fisica\Actividad 7.4\dataset_run")
RUTA_PERTURBADO = Path(r"C:\Users\ixoro\Documents\UNI\7moSemestre\Inteligencia Artificial Fisica\Actividad 7.4\dataset_perturbado_moderado")
CARPETA_RESULTADOS = RUTA_DATASET.parent / "resultados_cnn_moderado"
RUTA_MODELO = CARPETA_RESULTADOS / "modelo_cnn.keras"

CLASES = ["avanzar", "izquierda", "derecha", "obstaculo"]
TAM = 64
N_CALIBRACION = 300     # imágenes de train para calibrar la cuantización INT8
EXTENSIONES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

try:                                   # TensorFlow incluye el intérprete de TFLite
    Interpreter = tf.lite.Interpreter
except AttributeError:                 # versiones nuevas: paquete LiteRT
    from ai_edge_litert.interpreter import Interpreter


# ============================================================
# 1) DATOS (mismo preprocesamiento que en el entrenamiento)
# ============================================================
def preprocesar_imagen(ruta):
    with Image.open(ruta) as img:
        gris = img.convert("L")
        if gris.size != (TAM, TAM):
            gris = gris.resize((TAM, TAM), Image.LANCZOS)
        arreglo = np.asarray(gris, dtype=np.float32) / 255.0
    return arreglo[..., np.newaxis]


def cargar_carpeta(base):
    X, y = [], []
    for indice, clase in enumerate(CLASES):
        for ruta in sorted((base / clase).iterdir()):
            if ruta.suffix.lower() in EXTENSIONES:
                X.append(preprocesar_imagen(ruta))
                y.append(indice)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


print("Cargando imágenes...")
X_train, _ = cargar_carpeta(RUTA_DATASET / "train")
X_test, y_test = cargar_carpeta(RUTA_DATASET / "test")
hay_perturbado = RUTA_PERTURBADO.is_dir()
if hay_perturbado:
    X_pert, y_pert = cargar_carpeta(RUTA_PERTURBADO)


# ============================================================
# 2) MODELO DE INFERENCIA (sin capas de aumento de datos)
# ============================================================
# Las capas Random* y GaussianNoise solo actúan al entrenar; se quitan para el
# dispositivo embebido porque no aportan nada en inferencia y TFLite no siempre las soporta.
modelo = keras.models.load_model(RUTA_MODELO)
capas = [c for c in modelo.layers
         if c.name != "aumento"
         and not c.__class__.__name__.startswith("Random")
         and c.__class__.__name__ != "GaussianNoise"]
inferencia = keras.Sequential([keras.Input(shape=(TAM, TAM, 1))] + capas, name="cnn_inferencia")
inferencia.summary()
n_parametros = inferencia.count_params()


# ============================================================
# 3) CONVERSIÓN A TFLITE: FP32 e INT8
# ============================================================
def crear_convertidor():
    ruta_sm = CARPETA_RESULTADOS / "savedmodel_inferencia"
    try:
        inferencia.export(str(ruta_sm))                      # Keras 3
        return tf.lite.TFLiteConverter.from_saved_model(str(ruta_sm))
    except Exception:
        return tf.lite.TFLiteConverter.from_keras_model(inferencia)


# --- FP32 ---
conv = crear_convertidor()
tflite_fp32 = conv.convert()
ruta_fp32 = CARPETA_RESULTADOS / "modelo_fp32.tflite"
ruta_fp32.write_bytes(tflite_fp32)


# --- INT8 (cuantización entera completa) ---
def datos_calibracion():
    """La cuantización necesita ver ejemplos reales para fijar la escala de cada capa."""
    idx = np.random.default_rng(0).choice(len(X_train), min(N_CALIBRACION, len(X_train)), replace=False)
    for i in idx:
        yield [X_train[i:i + 1]]


conv = crear_convertidor()
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = datos_calibracion
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type = tf.int8       # entrada y salida también en INT8,
conv.inference_output_type = tf.int8      # como en un microcontrolador
tflite_int8 = conv.convert()
ruta_int8 = CARPETA_RESULTADOS / "modelo_int8.tflite"
ruta_int8.write_bytes(tflite_int8)


# ============================================================
# 4) MEDICIONES
# ============================================================
def crear_interprete(ruta):
    interp = Interpreter(model_path=str(ruta), num_threads=1)   # 1 hilo, como en un embebido
    interp.allocate_tensors()
    return interp


def estimar_ram(interp):
    """Estima la RAM de activaciones (el 'tensor arena' de TFLite Micro): el pico de memoria
    que necesitan al mismo tiempo las entradas y salidas de una capa. Los pesos no cuentan
    aquí porque se leen directamente desde la Flash."""
    try:
        ops = interp._get_ops_details()
    except AttributeError:
        return None
    tensores = {t["index"]: t for t in interp.get_tensor_details()}
    producidos = {i["index"] for i in interp.get_input_details()}
    for op in ops:
        producidos.update(op["outputs"])

    def bytes_tensor(i):
        t = tensores[i]
        return int(np.prod(t["shape"])) * np.dtype(t["dtype"]).itemsize

    pico = 0
    for op in ops:
        activos = {i for i in op["inputs"] if i in producidos} | set(op["outputs"])
        pico = max(pico, sum(bytes_tensor(i) for i in activos if i >= 0))
    return pico


def preparar_entrada(x, detalle):
    """Para INT8 hay que cuantizar la imagen: q = x / escala + punto_cero."""
    if detalle["dtype"] == np.int8:
        escala, cero = detalle["quantization"]
        return np.clip(np.round(x / escala + cero), -128, 127).astype(np.int8)
    return x.astype(np.float32)


def evaluar(interp, X, y=None, medir_tiempo=False):
    ent = interp.get_input_details()[0]
    sal = interp.get_output_details()[0]
    for x in X[:10]:                                  # calentamiento
        interp.set_tensor(ent["index"], preparar_entrada(x[np.newaxis], ent))
        interp.invoke()
    predicciones, tiempos = [], []
    for x in X:
        interp.set_tensor(ent["index"], preparar_entrada(x[np.newaxis], ent))
        t0 = time.perf_counter()
        interp.invoke()
        tiempos.append(time.perf_counter() - t0)
        predicciones.append(int(np.argmax(interp.get_tensor(sal["index"])[0])))
    acc = float(np.mean(np.array(predicciones) == y)) if y is not None else None
    return acc, float(np.mean(tiempos) * 1000)       # accuracy, ms por imagen


resultados = {}
for nombre, ruta, bytes_por_peso in [("FP32", ruta_fp32, 4), ("INT8", ruta_int8, 1)]:
    interp = crear_interprete(ruta)
    acc_test, ms = evaluar(interp, X_test, y_test)
    acc_pert = evaluar(interp, X_pert, y_pert)[0] if hay_perturbado else None
    resultados[nombre] = dict(
        tamano=n_parametros * bytes_por_peso,       # solo los pesos
        flash=ruta.stat().st_size,                  # archivo .tflite completo
        ram=estimar_ram(interp),
        ms=ms,
        acc=acc_test,
        acc_pert=acc_pert,
    )


# ============================================================
# 5) TABLA COMPARATIVA
# ============================================================
def kb(b):
    return "n/d" if b is None else f"{b / 1024:.1f} KB"


def reduccion(a, b):
    return "" if not a or b is None else f"{a / b:.1f}x menor"


fp, i8 = resultados["FP32"], resultados["INT8"]
filas = [
    ("Tamaño del modelo (pesos)", kb(fp["tamano"]), kb(i8["tamano"]), reduccion(fp["tamano"], i8["tamano"])),
    ("RAM (activaciones, estimada)", kb(fp["ram"]), kb(i8["ram"]), reduccion(fp["ram"], i8["ram"])),
    ("Flash (archivo .tflite)", kb(fp["flash"]), kb(i8["flash"]), reduccion(fp["flash"], i8["flash"])),
    ("Tiempo de inferencia", f"{fp['ms']:.3f} ms", f"{i8['ms']:.3f} ms",
     f"{fp['ms'] / i8['ms']:.2f}x" if i8["ms"] else ""),
    ("Accuracy (test)", f"{fp['acc']:.4f}", f"{i8['acc']:.4f}", f"{i8['acc'] - fp['acc']:+.4f}"),
]
if hay_perturbado:
    filas.append(("Accuracy (perturbado)", f"{fp['acc_pert']:.4f}", f"{i8['acc_pert']:.4f}",
                  f"{i8['acc_pert'] - fp['acc_pert']:+.4f}"))

print("\n" + "=" * 78)
print(f"COMPARACIÓN FP32 vs INT8   (parámetros: {n_parametros:,})")
print("=" * 78)
print(f"{'Característica':<31}{'FP32':>14}{'INT8':>14}{'Cambio':>17}")
print("-" * 78)
for f in filas:
    print(f"{f[0]:<31}{f[1]:>14}{f[2]:>14}{f[3]:>17}")
print("=" * 78)
print("Notas:")
print(" - El tiempo de inferencia se midió en esta PC con 1 hilo; en un microcontrolador")
print("   los tiempos son mayores, y la ventaja de INT8 suele ser más grande.")
print(" - El consumo de energía es aproximadamente proporcional a potencia x tiempo de")
print("   inferencia, y baja con INT8 por usar 4 veces menos memoria y operaciones enteras.")

with open(CARPETA_RESULTADOS / "comparacion_fp32_int8.csv", "w", encoding="utf-8") as f:
    f.write("Caracteristica,FP32,INT8,Cambio\n")
    for fila in filas:
        f.write(",".join(fila) + "\n")
print(f"\nModelos y tabla guardados en: {CARPETA_RESULTADOS}")