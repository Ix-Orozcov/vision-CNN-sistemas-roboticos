"""
CNN para clasificar imágenes en 4 categorías: avanzar, izquierda, derecha, obstaculo.
Al final evalúa la robustez con un conjunto de imágenes PERTURBADAS
(iluminación, distancia, ángulo de cámara y posición del patrón).

Estructura esperada:
    dataset/
        train/       avanzar/ izquierda/ derecha/ obstaculo/
        validation/  avanzar/ izquierda/ derecha/ obstaculo/
        test/        avanzar/ izquierda/ derecha/ obstaculo/
    dataset_perturbado/   avanzar/ izquierda/ derecha/ obstaculo/
        (generado con capturar_perturbado.py; archivos tipo 000_iluminacion.png)

Requisitos:
    pip install tensorflow scikit-learn matplotlib pillow numpy
"""

import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
    classification_report,
)

# ============================================================
# CONFIGURACIÓN (ajusta las rutas a tus carpetas)
# ============================================================
RUTA_DATASET = Path(r"C:\Users\ixoro\Documents\UNI\7moSemestre\Inteligencia Artificial Fisica\Actividad 7.4\dataset_run")
RUTA_PERTURBADO = Path(r"C:\Users\ixoro\Documents\UNI\7moSemestre\Inteligencia Artificial Fisica\Actividad 7.4\dataset_perturbado_severo")
CARPETA_RESULTADOS = RUTA_DATASET.parent / "resultados_cnn"

CLASES = ["avanzar", "izquierda", "derecha", "obstaculo"]
TAM = 64          # tamaño de entrada de la red (las imágenes de otro tamaño se redimensionan)
EPOCAS = 100
BATCH = 32
SEMILLA = 42
EXTENSIONES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

keras.utils.set_random_seed(SEMILLA)
CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1) CARGA, ESCALA DE GRISES Y NORMALIZACIÓN
# ============================================================
def preprocesar_imagen(ruta):
    """Abre una imagen, la pasa a escala de grises, la ajusta a 64x64 y la normaliza a [0, 1]."""
    with Image.open(ruta) as img:
        gris = img.convert("L")                      # escala de grises (1 canal)
        if gris.size != (TAM, TAM):
            gris = gris.resize((TAM, TAM), Image.LANCZOS)
        arreglo = np.asarray(gris, dtype=np.float32)
    arreglo = arreglo / 255.0                        # normalización 0-255 -> 0-1
    return arreglo[..., np.newaxis]                  # forma (64, 64, 1)


def cargar_carpeta(base, nombre):
    """Carga las imágenes de base/<clase>/ con sus etiquetas y rutas."""
    X, y, rutas = [], [], []
    for indice, clase in enumerate(CLASES):
        carpeta = base / clase
        if not carpeta.is_dir():
            raise FileNotFoundError(f"No se encontró la carpeta: {carpeta}")
        for ruta in sorted(carpeta.iterdir()):
            if ruta.suffix.lower() in EXTENSIONES:
                X.append(preprocesar_imagen(ruta))
                y.append(indice)
                rutas.append(ruta)
    if not X:
        raise FileNotFoundError(f"No hay imágenes en: {base}")
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    print(f"{nombre:<11} -> {len(X)} imágenes | por clase: "
          + ", ".join(f"{c}={np.sum(y == i)}" for i, c in enumerate(CLASES)))
    return X, y, rutas


print("Cargando imágenes (escala de grises + normalización)...")
X_train, y_train, _ = cargar_carpeta(RUTA_DATASET / "train", "train")
X_val, y_val, _ = cargar_carpeta(RUTA_DATASET / "validation", "validation")
X_test, y_test, _ = cargar_carpeta(RUTA_DATASET / "test", "test")
# Se carga desde el inicio para detectar una ruta incorrecta ANTES de entrenar
X_pert, y_pert, rutas_pert = cargar_carpeta(RUTA_PERTURBADO, "perturbado")
print(f"Rango de píxeles después de normalizar: [{X_train.min():.2f}, {X_train.max():.2f}]")


# ============================================================
# 2) RED NEURONAL CONVOLUCIONAL
# ============================================================
aumento = keras.Sequential([
    layers.RandomBrightness(0.25, value_range=(0, 1)),          # iluminación
    layers.RandomContrast(0.25, value_range=(0, 1)),            # iluminación
    layers.RandomZoom(height_factor=0.2, width_factor=(-0.1, 0.35), fill_mode="nearest"),                # distancia
    layers.RandomTranslation(0.1, 0.1, fill_mode="nearest"),    # posición del patrón
    layers.RandomRotation(0.04, fill_mode="nearest"),           # ángulo (±14°)
], name="aumento")

modelo = keras.Sequential([
    keras.Input(shape=(TAM, TAM, 1)),
    aumento,
    layers.Conv2D(8, (3, 3), activation="relu", name="conv1"),
    layers.MaxPooling2D((2, 2), name="pool1"),
    layers.Conv2D(16, (3, 3), activation="relu", name="conv2"),
    layers.MaxPooling2D((2, 2), name="pool2"),
    layers.Flatten(name="flatten"),
    layers.Dense(16, activation="relu", name="dense"),
    layers.Dense(len(CLASES), activation="softmax", name="salida"),
])

modelo.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
modelo.summary()


# ============================================================
# 3) ENTRENAMIENTO (midiendo el tiempo)
# ============================================================
inicio = time.perf_counter()

parada = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=15,
                                       restore_best_weights=True)

historial = modelo.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCAS,
    batch_size=BATCH,
    shuffle=True,
    callbacks=[parada],
    verbose=2,
)
tiempo_entrenamiento = time.perf_counter() - inicio

modelo.save(CARPETA_RESULTADOS / "modelo_cnn.keras")


# ============================================================
# 4) PARÁMETROS DEL ENTRENAMIENTO
# ============================================================
loss_train, acc_train = modelo.evaluate(X_train, y_train, verbose=0)
loss_val, acc_val = modelo.evaluate(X_val, y_val, verbose=0)
loss_test, acc_test = modelo.evaluate(X_test, y_test, verbose=0)

print("\n" + "=" * 45)
print("RESULTADOS DEL ENTRENAMIENTO")
print("=" * 45)
print(f"Accuracy entrenamiento : {acc_train:.4f}")
print(f"Accuracy validación    : {acc_val:.4f}")
print(f"Loss entrenamiento     : {loss_train:.4f}")
print(f"Loss validación        : {loss_val:.4f}")
print(f"Número de parámetros   : {modelo.count_params():,}")
print(f"Tiempo de entrenamiento: {tiempo_entrenamiento:.2f} s")
print(f"(Accuracy en test      : {acc_test:.4f} | Loss test: {loss_test:.4f})")
print("=" * 45)


# ============================================================
# 5) INFERENCIA SOBRE EL CONJUNTO DE TEST
# ============================================================
probabilidades = modelo.predict(X_test, verbose=0)
y_pred = np.argmax(probabilidades, axis=1)

print("\nReporte de clasificación (test):")
print(classification_report(y_test, y_pred, labels=range(len(CLASES)),
                            target_names=CLASES, digits=4, zero_division=0))


def predecir_imagen(ruta):
    """Inferencia para una sola imagen nueva."""
    x = preprocesar_imagen(ruta)[np.newaxis, ...]
    prob = modelo.predict(x, verbose=0)[0]
    return CLASES[int(np.argmax(prob))], float(np.max(prob))


def mostrar_ejemplos(X, y_real, y_predicha, prob, titulo, archivo):
    """Cuadrícula con imágenes, su clase real y la predicha (verde = acierto, rojo = error)."""
    n = min(12, len(X))
    indices = np.random.default_rng(SEMILLA).choice(len(X), n, replace=False)
    fig, ejes = plt.subplots(2, 6, figsize=(14, 5.5))
    for eje in ejes.flat:
        eje.axis("off")
    for eje, i in zip(ejes.flat, indices):
        correcto = y_predicha[i] == y_real[i]
        eje.imshow(X[i].squeeze(), cmap="gray", vmin=0, vmax=1)
        eje.set_title(f"Real: {CLASES[y_real[i]]}\nPred: {CLASES[y_predicha[i]]} "
                      f"({prob[i].max():.0%})",
                      fontsize=9, color="green" if correcto else "red")
    fig.suptitle(titulo)
    fig.tight_layout()
    fig.savefig(CARPETA_RESULTADOS / archivo, dpi=150)


mostrar_ejemplos(X_test, y_test, y_pred, probabilidades,
                 "Inferencia sobre imágenes de test", "inferencia_ejemplos.png")


# ============================================================
# 6) GRÁFICAS
# ============================================================
# --- Curvas de aprendizaje ---
epocas = range(1, len(historial.history["loss"]) + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
ax1.plot(epocas, historial.history["accuracy"], "o-", label="Entrenamiento")
ax1.plot(epocas, historial.history["val_accuracy"], "s-", label="Validación")
ax1.set_title("Accuracy por época")
ax1.set_xlabel("Época")
ax1.set_ylabel("Accuracy")
ax1.grid(alpha=0.3)
ax1.legend()
ax2.plot(epocas, historial.history["loss"], "o-", label="Entrenamiento")
ax2.plot(epocas, historial.history["val_loss"], "s-", label="Validación")
ax2.set_title("Loss por época")
ax2.set_xlabel("Época")
ax2.set_ylabel("Loss")
ax2.grid(alpha=0.3)
ax2.legend()
fig.suptitle("Curvas de aprendizaje")
fig.tight_layout()
fig.savefig(CARPETA_RESULTADOS / "curvas_aprendizaje.png", dpi=150)


def graficar_matriz_confusion(y_real, y_predicha, titulo, archivo):
    cm = confusion_matrix(y_real, y_predicha, labels=range(len(CLASES)))
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ConfusionMatrixDisplay(cm, display_labels=CLASES).plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title(titulo)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Clase real")
    fig.tight_layout()
    fig.savefig(CARPETA_RESULTADOS / archivo, dpi=150)
    return cm


def accuracy_por_clase(cm):
    totales = cm.sum(axis=1)
    return np.divide(np.diag(cm), totales, out=np.zeros(len(CLASES)), where=totales > 0)


# --- Matriz de confusión ---
cm = graficar_matriz_confusion(y_test, y_pred, "Matriz de confusión (test)", "matriz_confusion.png")

# --- Métricas por clase ---
accuracy_clase = accuracy_por_clase(cm)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, y_pred, labels=range(len(CLASES)), zero_division=0)


def grafica_barras(valores, titulo, archivo, color):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    barras = ax.bar(CLASES, valores, color=color, edgecolor="black")
    for barra, v in zip(barras, valores):
        ax.text(barra.get_x() + barra.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Valor")
    ax.set_title(titulo)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CARPETA_RESULTADOS / archivo, dpi=150)


grafica_barras(accuracy_clase, "Accuracy por clase (test)", "accuracy_por_clase.png", "#4C72B0")
grafica_barras(precision, "Precision por clase (test)", "precision.png", "#55A868")
grafica_barras(recall, "Recall por clase (test)", "recall.png", "#C44E52")
grafica_barras(f1, "F1-score por clase (test)", "f1_score.png", "#8172B2")


# ============================================================
# 7) EVALUACIÓN DE ROBUSTEZ CON IMÁGENES PERTURBADAS
# ============================================================
prob_pert = modelo.predict(X_pert, verbose=0)
y_pred_pert = np.argmax(prob_pert, axis=1)

acc_nominal = float(np.mean(y_pred == y_test))          # test original (condiciones nominales)
acc_perturbado = float(np.mean(y_pred_pert == y_pert))  # nuevas imágenes perturbadas
caida = acc_nominal - acc_perturbado

print("\n" + "=" * 55)
print("EVALUACIÓN DE ROBUSTEZ")
print("=" * 55)
print(f"Accuracy_nominal    (test, {len(y_test)} imágenes)      : {acc_nominal:.4f}")
print(f"Accuracy_perturbado (perturbado, {len(y_pert)} imágenes): {acc_perturbado:.4f}")
print(f"\nAccuracy_nominal - Accuracy_perturbado = "
      f"{acc_nominal:.4f} - {acc_perturbado:.4f} = {caida:.4f}")
print(f"  -> la accuracy {'baja' if caida >= 0 else 'sube'} "
      f"{abs(caida) * 100:.1f} puntos porcentuales con las perturbaciones")

# Desglose por tipo de perturbación (se toma del nombre del archivo: 000_iluminacion.png)
tipos = np.array([r.stem.split("_", 1)[1] if "_" in r.stem else "sin_tipo" for r in rutas_pert])
nombres_tipos = sorted(set(tipos))
acc_tipos = []
print(f"\n{'Perturbación':<18}{'Imágenes':>9}{'Accuracy':>10}{'Caída':>9}")
for t in nombres_tipos:
    m = tipos == t
    acc_t = float(np.mean(y_pred_pert[m] == y_pert[m]))
    acc_tipos.append(acc_t)
    print(f"{t:<18}{m.sum():>9}{acc_t:>10.4f}{acc_nominal - acc_t:>9.4f}")
print("=" * 55)

print("\nReporte de clasificación (perturbado):")
print(classification_report(y_pert, y_pred_pert, labels=range(len(CLASES)),
                            target_names=CLASES, digits=4, zero_division=0))

# --- Gráfica: accuracy nominal vs. cada perturbación ---
etiquetas = ["nominal (test)", "perturbado (total)"] + nombres_tipos
valores = [acc_nominal, acc_perturbado] + acc_tipos
colores = ["#4C72B0", "#C44E52"] + ["#DD8452"] * len(nombres_tipos)
fig, ax = plt.subplots(figsize=(9, 4.8))
barras = ax.bar(etiquetas, valores, color=colores, edgecolor="black")
for barra, v in zip(barras, valores):
    ax.text(barra.get_x() + barra.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
ax.set_ylim(0, 1.1)
ax.set_ylabel("Accuracy")
ax.set_title(f"Robustez: Accuracy_nominal - Accuracy_perturbado = {caida:.3f}")
ax.grid(axis="y", alpha=0.3)
plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
fig.tight_layout()
fig.savefig(CARPETA_RESULTADOS / "robustez_perturbaciones.png", dpi=150)

# --- Accuracy por clase: nominal vs. perturbado ---
cm_pert = graficar_matriz_confusion(y_pert, y_pred_pert, "Matriz de confusión (perturbado)",
                                    "matriz_confusion_perturbado.png")
acc_clase_pert = accuracy_por_clase(cm_pert)
x = np.arange(len(CLASES))
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(x - 0.2, accuracy_clase, 0.4, label="Nominal (test)", color="#4C72B0", edgecolor="black")
ax.bar(x + 0.2, acc_clase_pert, 0.4, label="Perturbado", color="#C44E52", edgecolor="black")
ax.set_xticks(x, CLASES)
ax.set_ylim(0, 1.1)
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy por clase: nominal vs. perturbado")
ax.grid(axis="y", alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(CARPETA_RESULTADOS / "accuracy_clase_nominal_vs_perturbado.png", dpi=150)

mostrar_ejemplos(X_pert, y_pert, y_pred_pert, prob_pert,
                 "Inferencia sobre imágenes perturbadas", "inferencia_perturbado.png")

print(f"\nGráficas y modelo guardados en: {CARPETA_RESULTADOS}")

# Ejemplo de uso para una imagen nueva:
# clase, confianza = predecir_imagen(r"C:\ruta\a\una\imagen.png")
# print(f"Predicción: {clase} ({confianza:.1%})")

plt.show()