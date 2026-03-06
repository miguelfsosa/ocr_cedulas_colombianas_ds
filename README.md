# OCR Cedulas Colombianas

Pipeline modular de extraccion automatica de campos en cedulas de ciudadania colombianas usando OCR y vision por computadora.

## Descripcion

Sistema que recibe una fotografia de cedula colombiana y retorna un JSON estructurado con numero de cedula, apellidos y nombres. El pipeline aplica preprocesamiento de imagen, evaluacion de calidad, segmentacion por template y reconocimiento optico de caracteres.

**Referencia academica:** Oliveira et al. (2020) — precision de referencia >93% en tareas KYC.

## Arquitectura del Pipeline

```
Foto de cedula (JPG/PNG)
        |
[1] Preprocesamiento (OpenCV)
    Redimensionar → deteccion de bordes → correccion de perspectiva → CLAHE → denoising
        |
[2] Evaluador de Calidad (ResNet-18 / heuristico)
    Score de calidad + orientacion. Rechaza imagenes de baja calidad.
        |
[3] Segmentador de Campos (template-based)
    Divide la cedula en ROIs: foto | apellidos | nombres | numero | fechas | firma
        |
[4] OCR por Region (EasyOCR)
    Reconocimiento en español + ingles, con score de confianza por campo
        |
[5] Extractor y Validador
    Regex para numero y fechas. Heuristica para nombres. Validacion global.
        |
JSON estructurado + imagen anotada (opcional)
```

## Estructura del Repositorio

```
deeplearning_banking/
├── src/kyc_ocr/
│   ├── __init__.py              # Exporta KYCPipeline
│   ├── preprocesamiento.py      # Modulo 1: preprocesamiento OpenCV
│   ├── evaluador_calidad.py     # Modulo 2: evaluador ResNet-18 / heuristico
│   ├── segmentador_campos.py    # Modulo 3: segmentacion por template
│   ├── extractor_campos.py      # Modulo 5: regex + validacion
│   └── pipeline.py              # Orquestador principal (KYCPipeline)
├── scripts/
│   └── procesar_cedula.py       # CLI de linea de comandos
├── data/kyc/
│   ├── muestras/                # Imagenes de prueba (no incluidas en repo)
│   └── resultados/              # JSONs y anotaciones de salida (generado en runtime)
├── docs/
│   ├── kyc_ocr_arquitectura.md  # Documentacion tecnica detallada
│   └── kyc_ocr_explicacion_tecnica.md
├── lecturas/                    # Material academico fuente
├── requirements.txt
└── README.md
```

## Instalacion

```bash
git clone https://github.com/miguelfsosa/ocr_cedulas_colombianas_ds.git
cd ocr_cedulas_colombianas_ds
pip install -r requirements.txt
```

> **Nota:** EasyOCR descarga los modelos (~95 MB) en el primer uso automaticamente.

## Uso

### CLI

```bash
# Procesamiento basico
python scripts/procesar_cedula.py --imagen data/kyc/muestras/cedula.jpg

# Guardar JSON de resultado
python scripts/procesar_cedula.py --imagen cedula.jpg --guardar-json

# Ver anotaciones visuales y detalles
python scripts/procesar_cedula.py --imagen cedula.jpg --mostrar-anotaciones --verbose
```

### Python

```python
from src.kyc_ocr import KYCPipeline

pipeline = KYCPipeline()
resultado = pipeline.procesar("ruta/cedula.jpg")

print(resultado["campos"]["numero"]["valor"])
print(resultado["campos"]["apellidos"]["valor"])
print(resultado["campos"]["nombres"]["valor"])
```

### Ejemplo de salida JSON

```json
{
  "estado": "procesado",
  "calidad": {
    "calidad_score": 0.82,
    "aceptable": true,
    "modo": "heuristico"
  },
  "campos": {
    "numero": { "valor": "12345678", "confianza": 0.95 },
    "apellidos": { "valor": "PEREZ GARCIA", "confianza": 0.91 },
    "nombres": { "valor": "JUAN CARLOS", "confianza": 0.90 },
    "_meta": { "valido": true, "confianza_global": 0.92 }
  },
  "tiempo_s": 3.42
}
```

## Dependencias Principales

| Paquete | Version minima | Uso |
|---|---|---|
| opencv-python | 4.8.0 | Preprocesamiento de imagen |
| easyocr | 1.7.0 | Motor OCR |
| torch | 2.0.0 | Backbone ResNet-18 |
| torchvision | 0.15.0 | Modelos preentrenados |
| Pillow | 10.0.0 | Conversion de imagen |
| numpy | 1.24.0 | Operaciones matriciales |

## Notas

- El evaluador de calidad opera en **modo heuristico** por defecto (sin GPU). Se puede proveer un modelo ResNet-18 fine-tuned con `--modelo-calidad`.
- Las proporciones del template (`segmentador_campos.py`) son configurables para adaptar el pipeline a otros tipos de documento.
- Las imagenes de cedulas **no se incluyen** en el repositorio por privacidad.
