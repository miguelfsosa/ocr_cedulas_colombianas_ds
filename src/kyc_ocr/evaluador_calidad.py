"""
Módulo 2: Evaluador de calidad de imagen con ResNet-18.

En ausencia de modelo fine-tuned, opera en modo heurístico:
  - Varianza de Laplacian para detectar desenfoque
  - Histograma para detectar sub/sobreexposición
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

try:
    from torchvision import models, transforms
    TORCHVISION_DISPONIBLE = True
except ImportError:
    TORCHVISION_DISPONIBLE = False

# Umbrales heurísticos
UMBRAL_BLUR = 80.0       # varianza Laplacian mínima
UMBRAL_CALIDAD = 0.5     # score mínimo para aceptar imagen
ORIENTACIONES = [0, 90, 180, 270]


def _varianza_laplacian(img_gris: np.ndarray) -> float:
    """Métrica de nitidez: mayor valor = imagen más nítida."""
    return float(cv2.Laplacian(img_gris, cv2.CV_64F).var())


def _score_exposicion(img_gris: np.ndarray) -> float:
    """
    Score de exposición [0,1]: penaliza imágenes muy oscuras o muy brillantes.
    """
    hist = cv2.calcHist([img_gris], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()
    # Porcentaje de píxeles en extremos (muy oscuro <20 o muy claro >235)
    extremos = hist[:20].sum() + hist[235:].sum()
    return float(1.0 - extremos)


def _heuristico(img_bgr: np.ndarray) -> dict:
    """Evaluación heurística sin modelo neural."""
    gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur_score = _varianza_laplacian(gris)
    expo_score = _score_exposicion(gris)

    blur_norm = min(blur_score / 300.0, 1.0)
    calidad_score = 0.6 * blur_norm + 0.4 * expo_score
    aceptable = blur_score >= UMBRAL_BLUR and calidad_score >= UMBRAL_CALIDAD

    return {
        "calidad_score": round(calidad_score, 4),
        "varianza_laplacian": round(blur_score, 2),
        "score_exposicion": round(expo_score, 4),
        "orientacion_grados": 0,
        "aceptable": aceptable,
        "modo": "heuristico",
    }


class _CabeceraCalidad(nn.Module):
    """Cabecera de clasificación sobre ResNet-18 (512 → 5 clases)."""

    def __init__(self):
        super().__init__()
        self.red = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 5),  # 4 orientaciones + rechazo
        )

    def forward(self, x):
        return self.red(x)


class EvaluadorCalidad:
    """
    Evalúa calidad y orientación de imagen de cédula.

    Si se proporciona ruta_modelo, carga pesos fine-tuned.
    Si no, usa modo heurístico.
    """

    def __init__(self, ruta_modelo: str | Path | None = None):
        self.dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.modelo = None
        self.transform = None

        if ruta_modelo and Path(ruta_modelo).exists() and TORCHVISION_DISPONIBLE:
            self._cargar_modelo(ruta_modelo)

    def _cargar_modelo(self, ruta_modelo: str | Path):
        backbone = models.resnet18(weights=None)
        backbone.fc = _CabeceraCalidad()
        backbone.load_state_dict(torch.load(str(ruta_modelo),
                                            map_location=self.dispositivo))
        backbone.eval()
        self.modelo = backbone.to(self.dispositivo)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def evaluar(self, img_bgr: np.ndarray) -> dict:
        """
        Evalúa la imagen.
        Retorna dict con: calidad_score, orientacion_grados, aceptable, modo.
        """
        if self.modelo is None:
            return _heuristico(img_bgr)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(img_rgb).unsqueeze(0).to(self.dispositivo)

        with torch.no_grad():
            logits = self.modelo(tensor)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        clase = int(np.argmax(probs))
        # Clase 4 = rechazo; 0-3 = orientaciones 0/90/180/270
        aceptable = clase < 4
        orientacion = ORIENTACIONES[clase] if aceptable else 0
        calidad_score = float(probs[clase]) if aceptable else float(probs[4])

        return {
            "calidad_score": round(float(np.max(probs[:4])), 4),
            "orientacion_grados": orientacion,
            "aceptable": aceptable,
            "modo": "resnet18",
        }
