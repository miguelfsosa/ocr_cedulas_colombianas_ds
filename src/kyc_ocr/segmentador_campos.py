"""
Modulo 3: Segmentador de campos basado en proporciones relativas (template).

Las proporciones estan normalizadas al tamano del documento (alto=1, ancho=1).
Cada entrada: (y_inicio, y_fin, x_inicio, x_fin) en [0, 1].
"""

import cv2
import numpy as np


# Template generico (referencia, no usado por defecto)
TEMPLATE_CEDULA = {
    "foto":              (0.10, 0.80, 0.03, 0.30),
    "nombre":            (0.10, 0.30, 0.33, 0.97),
    "numero":            (0.30, 0.50, 0.33, 0.97),
    "fecha_nacimiento":  (0.50, 0.68, 0.33, 0.75),
    "fecha_expedicion":  (0.50, 0.68, 0.75, 0.97),
    "firma":             (0.72, 0.95, 0.33, 0.97),
}

# Cedula de Ciudadania colombiana (formato plastico moderno, anverso)
# Campos: numero | apellidos | nombres | firma | foto
# Proporciones medidas sobre imagenes reales
TEMPLATE_CEDULA_COLOMBIANA = {
    "foto":      (0.13, 0.87, 0.60, 0.97),
    "numero":    (0.18, 0.37, 0.02, 0.60),
    "apellidos": (0.35, 0.55, 0.02, 0.60),
    "nombres":   (0.53, 0.72, 0.02, 0.60),
    "firma":     (0.70, 0.90, 0.02, 0.55),
}


def _recortar_roi(img: np.ndarray, y0: float, y1: float,
                  x0: float, x1: float) -> np.ndarray:
    alto, ancho = img.shape[:2]
    yi, yf = int(y0 * alto), int(y1 * alto)
    xi, xf = int(x0 * ancho), int(x1 * ancho)
    return img[yi:yf, xi:xf].copy()


def segmentar(img: np.ndarray,
              template: dict | None = None) -> dict:
    """
    Divide la imagen en regiones de interes (ROI) segun las proporciones del template.

    Args:
        img: imagen BGR del documento ya preprocesado y rectificado.
        template: proporciones personalizadas (usa TEMPLATE_CEDULA_COLOMBIANA por defecto).

    Returns:
        Dict {nombre_campo: imagen_roi (BGR)}
    """
    t = template if template is not None else TEMPLATE_CEDULA_COLOMBIANA
    rois = {}
    for campo, (y0, y1, x0, x1) in t.items():
        roi = _recortar_roi(img, y0, y1, x0, x1)
        rois[campo] = roi
    return rois


def dibujar_regiones(img: np.ndarray,
                     template: dict | None = None) -> np.ndarray:
    """Dibuja los rectangulos de cada campo sobre la imagen (depuracion visual)."""
    t = template if template is not None else TEMPLATE_CEDULA_COLOMBIANA
    copia = img.copy()
    alto, ancho = img.shape[:2]
    colores = {
        "foto":      (255, 100,   0),
        "numero":    (  0, 100, 255),
        "apellidos": (  0, 200,   0),
        "nombres":   (  0, 180, 100),
        "firma":     (100, 100, 100),
        # template generico
        "nombre":           (0, 200, 0),
        "fecha_nacimiento": (200, 0, 200),
        "fecha_expedicion": (0, 200, 200),
    }
    for campo, (y0, y1, x0, x1) in t.items():
        yi, yf = int(y0 * alto), int(y1 * alto)
        xi, xf = int(x0 * ancho), int(x1 * ancho)
        color = colores.get(campo, (200, 200, 200))
        cv2.rectangle(copia, (xi, yi), (xf, yf), color, 2)
        cv2.putText(copia, campo, (xi + 4, yi + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
                    cv2.LINE_AA)
    return copia
