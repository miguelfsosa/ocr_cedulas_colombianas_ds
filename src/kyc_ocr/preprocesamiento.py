"""
Módulo 1: Preprocesamiento de imágenes de cédula nacional.

Pipeline:
  redimensionar → detectar_documento → corregir_perspectiva
  → mejorar_contraste → reducir_ruido
"""

import cv2
import numpy as np
from pathlib import Path


def redimensionar(img: np.ndarray, max_ancho: int = 1280) -> np.ndarray:
    """Redimensiona la imagen manteniendo proporción si excede max_ancho."""
    alto, ancho = img.shape[:2]
    if ancho <= max_ancho:
        return img
    factor = max_ancho / ancho
    nuevo_alto = int(alto * factor)
    return cv2.resize(img, (max_ancho, nuevo_alto), interpolation=cv2.INTER_AREA)


def _ordenar_esquinas(pts: np.ndarray) -> np.ndarray:
    """Ordena 4 puntos: arriba-izq, arriba-der, abajo-der, abajo-izq."""
    rect = np.zeros((4, 2), dtype="float32")
    sumas = pts.sum(axis=1)
    rect[0] = pts[np.argmin(sumas)]   # arriba-izq: menor suma
    rect[2] = pts[np.argmax(sumas)]   # abajo-der: mayor suma
    diffs = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diffs)]   # arriba-der: menor diferencia
    rect[3] = pts[np.argmax(diffs)]   # abajo-izq: mayor diferencia
    return rect


def detectar_documento(img: np.ndarray) -> np.ndarray | None:
    """
    Detecta los 4 bordes del documento usando contornos.
    Retorna array (4,2) con las esquinas o None si no se detecta.
    """
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gris = cv2.GaussianBlur(gris, (5, 5), 0)
    bordes = cv2.Canny(gris, 75, 200)

    contornos, _ = cv2.findContours(bordes, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)[:5]

    for c in contornos:
        perimetro = cv2.arcLength(c, True)
        aprox = cv2.approxPolyDP(c, 0.02 * perimetro, True)
        if len(aprox) == 4:
            return _ordenar_esquinas(aprox.reshape(4, 2).astype("float32"))

    return None


def corregir_perspectiva(img: np.ndarray, esquinas: np.ndarray) -> np.ndarray:
    """
    Aplica transformación de perspectiva para obtener el documento recto.
    Si esquinas es None, retorna la imagen original.
    """
    if esquinas is None:
        return img

    (ai, ad, abd, abizq) = esquinas
    ancho_a = np.linalg.norm(ad - ai)
    ancho_b = np.linalg.norm(abd - abizq)
    ancho_max = max(int(ancho_a), int(ancho_b))

    alto_a = np.linalg.norm(abizq - ai)
    alto_b = np.linalg.norm(abd - ad)
    alto_max = max(int(alto_a), int(alto_b))

    destino = np.array([
        [0, 0],
        [ancho_max - 1, 0],
        [ancho_max - 1, alto_max - 1],
        [0, alto_max - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(esquinas, destino)
    return cv2.warpPerspective(img, M, (ancho_max, alto_max))


def mejorar_contraste(img: np.ndarray) -> np.ndarray:
    """Aplica CLAHE sobre el canal L del espacio LAB."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def reducir_ruido(img: np.ndarray) -> np.ndarray:
    """Aplica filtro de reducción de ruido no local para imagen color."""
    return cv2.fastNlMeansDenoisingColored(img, None, h=10, hColor=10,
                                           templateWindowSize=7,
                                           searchWindowSize=21)


def preprocesar(ruta_imagen: str | Path) -> np.ndarray:
    """
    Pipeline completo de preprocesamiento.
    Retorna imagen BGR lista para OCR.
    """
    ruta = Path(ruta_imagen)
    if not ruta.exists():
        raise FileNotFoundError(f"Imagen no encontrada: {ruta}")

    img = cv2.imread(str(ruta))
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {ruta}")

    img = redimensionar(img)
    esquinas = detectar_documento(img)
    img = corregir_perspectiva(img, esquinas)
    img = mejorar_contraste(img)
    img = reducir_ruido(img)
    return img
