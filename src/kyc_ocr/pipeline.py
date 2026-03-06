"""
Modulo orquestador: KYCPipeline

Encadena los 5 modulos del sistema OCR para cedula colombiana.
"""

import time
import json
import cv2
import numpy as np
from pathlib import Path

from .preprocesamiento import preprocesar
from .evaluador_calidad import EvaluadorCalidad
from .segmentador_campos import segmentar, dibujar_regiones
from .extractor_campos import (
    extraer_numero_cedula,
    extraer_nombre,
    validar_resultado,
)

try:
    import easyocr
    _READER = None

    def _get_reader():
        global _READER
        if _READER is None:
            _READER = easyocr.Reader(['es', 'en'], verbose=False)
        return _READER

    EASYOCR_DISPONIBLE = True
except ImportError:
    EASYOCR_DISPONIBLE = False

    def _get_reader():
        raise ImportError("EasyOCR no esta instalado. Ejecuta: pip install easyocr")


def _ocr_roi(roi: np.ndarray) -> tuple[str, float]:
    """Ejecuta OCR sobre una ROI. Retorna (texto_unido, confianza_promedio)."""
    if not EASYOCR_DISPONIBLE:
        return "", 0.0
    reader = _get_reader()
    resultados = reader.readtext(roi, detail=1)
    if not resultados:
        return "", 0.0
    textos = [r[1] for r in resultados]
    confianzas = [r[2] for r in resultados]
    return " ".join(textos), round(sum(confianzas) / len(confianzas), 4)


class KYCPipeline:
    """
    Pipeline completo de KYC para cedula de ciudadania colombiana.

    Args:
        ruta_modelo_calidad: pesos fine-tuned para ResNet-18 (opcional).
        umbral_calidad: score minimo [0,1] para aceptar la imagen (default 0.25).
                        El evaluador heuristico puede ser conservador con fondos
                        complejos; bajar este umbral si las imagenes tienen fondo.
    """

    def __init__(self, ruta_modelo_calidad: str | Path | None = None,
                 umbral_calidad: float = 0.25):
        self.evaluador = EvaluadorCalidad(ruta_modelo=ruta_modelo_calidad)
        self.umbral_calidad = umbral_calidad

    def procesar(self, ruta_imagen: str | Path,
                 guardar_anotacion: str | Path | None = None) -> dict:
        """
        Procesa una imagen de cedula y extrae campos estructurados.

        Returns:
            Dict con: estado, calidad, campos (numero, apellidos, nombres), tiempo_s
        """
        t_inicio = time.time()
        ruta = Path(ruta_imagen)

        # Modulo 1: Preprocesamiento
        img = preprocesar(ruta)

        # Modulo 2: Evaluacion de calidad
        # Se usa calidad_score directamente para que umbral_calidad del pipeline
        # tenga efecto real, independientemente del umbral interno del evaluador.
        calidad = self.evaluador.evaluar(img)
        score = calidad.get("calidad_score", 0.0)
        if score < self.umbral_calidad:
            return {
                "estado": "rechazado",
                "motivo": f"Imagen de baja calidad (score={score:.4f} < umbral={self.umbral_calidad})",
                "calidad": calidad,
                "tiempo_s": round(time.time() - t_inicio, 2),
            }

        # Modulo 3: Segmentacion de campos (template colombiano por defecto)
        rois = segmentar(img)

        # Modulo 4: OCR por region (excepto foto)
        textos_ocr: dict[str, tuple[str, float]] = {}
        for campo, roi in rois.items():
            if campo == "foto":
                continue
            texto, confianza = _ocr_roi(roi)
            textos_ocr[campo] = (texto, confianza)

        # Modulo 5: Extraccion y validacion
        campos = {}

        texto_num, conf_num = textos_ocr.get("numero", ("", 0.0))
        campos["numero"] = {
            "valor": extraer_numero_cedula(texto_num),
            "texto_ocr": texto_num,
            "confianza": conf_num,
        }

        texto_ap, conf_ap = textos_ocr.get("apellidos", ("", 0.0))
        campos["apellidos"] = {
            "valor": extraer_nombre(texto_ap),
            "texto_ocr": texto_ap,
            "confianza": conf_ap,
        }

        texto_nom, conf_nom = textos_ocr.get("nombres", ("", 0.0))
        campos["nombres"] = {
            "valor": extraer_nombre(texto_nom),
            "texto_ocr": texto_nom,
            "confianza": conf_nom,
        }

        campos = validar_resultado(campos)

        # Anotacion visual opcional
        if guardar_anotacion:
            img_anotada = dibujar_regiones(img)
            cv2.imwrite(str(guardar_anotacion), img_anotada)

        return {
            "estado": "procesado",
            "imagen": str(ruta),
            "calidad": calidad,
            "campos": campos,
            "tiempo_s": round(time.time() - t_inicio, 2),
        }

    def procesar_a_json(self, ruta_imagen: str | Path,
                        ruta_salida: str | Path | None = None,
                        guardar_anotacion: str | Path | None = None) -> str:
        """Como procesar(), pero retorna y opcionalmente guarda JSON."""
        resultado = self.procesar(ruta_imagen, guardar_anotacion=guardar_anotacion)
        json_str = json.dumps(resultado, ensure_ascii=False, indent=2)
        if ruta_salida:
            Path(ruta_salida).parent.mkdir(parents=True, exist_ok=True)
            Path(ruta_salida).write_text(json_str, encoding="utf-8")
        return json_str
