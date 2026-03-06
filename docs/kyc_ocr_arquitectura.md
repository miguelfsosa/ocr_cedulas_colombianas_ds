# Arquitectura: Sistema OCR para Documentos de Identidad (Cedula Nacional)

## Contexto

Pipeline modular que extrae campos estructurados (nombre, numero de cedula,
fechas) de fotografias de cedulas nacionales tomadas en condiciones reales.

Referencia academica: Oliveira et al. (2020), precision de referencia >93% en KYC.

---

## Diagrama del Pipeline

```
Foto de cedula (JPG/PNG)
        |
[1] Preprocesamiento (OpenCV)
    - Redimensionar a maximo 1280px
    - Deteccion de bordes del documento (contornos)
    - Correccion de perspectiva (warpPerspective)
    - Mejora de contraste: CLAHE sobre canal L (espacio LAB)
    - Reduccion de ruido: fastNlMeansDenoisingColored
        |
[2] Evaluador de Calidad CNN (ResNet-18)
    - Entrada: imagen preprocesada (224x224)
    - Salida: score de calidad + orientacion detectada
    - Modo heuristico si no hay modelo fine-tuned (Laplacian + histograma)
    - Si score < umbral: rechazar imagen con mensaje al usuario
        |
[3] Segmentador de Campos (template-based)
    - Divide la cedula en ROIs por proporciones relativas:
      foto | nombre | numero | fecha_nacimiento | fecha_expedicion | firma
        |
[4] Reconocimiento de Texto (EasyOCR)
    - OCR region por region (excepto foto)
    - Idiomas: espanol + ingles
    - Retorna lista de (texto, confianza) por region
        |
[5] Extractor y Validador de Campos
    - Regex para numero de cedula, fechas
    - Heuristica para nombres en mayusculas
    - Confianza agregada por campo
        |
JSON estructurado + imagen anotada (opcional)
```

---

## Estructura de Archivos

```
src/kyc_ocr/
    __init__.py             — exporta KYCPipeline
    preprocesamiento.py     — Modulo 1: OpenCV preprocessing
    evaluador_calidad.py    — Modulo 2: ResNet-18 quality scorer
    segmentador_campos.py   — Modulo 3: segmentacion por template
    extractor_campos.py     — Modulo 5: regex + validacion
    pipeline.py             — orquestador principal (KYCPipeline)

scripts/
    procesar_cedula.py      — CLI de linea de comandos

data/kyc/
    muestras/               — imagenes de prueba (aportadas por el usuario)
    resultados/             — JSONs y anotaciones de salida (creado en runtime)

docs/
    kyc_ocr_arquitectura.md — este archivo
```

---

## Dependencias

| Paquete         | Version minima | Uso                          |
|-----------------|----------------|------------------------------|
| opencv-python   | 4.8.0          | Preprocesamiento de imagen   |
| easyocr         | 1.7.0          | Motor OCR                    |
| torch           | 2.0.0          | Backbone ResNet-18           |
| torchvision     | 0.15.0         | Modelos preentrenados        |
| Pillow          | 10.0.0         | Conversion de imagen         |
| numpy           | 1.24.0         | Operaciones matriciales      |

Instalar todo:
```bash
pip install -r requirements.txt
```

---

## Como ejecutar el CLI

### Uso basico

```bash
python scripts/procesar_cedula.py --imagen data/kyc/muestras/cedula.jpg
```

### Guardar JSON de resultado

```bash
python scripts/procesar_cedula.py --imagen cedula.jpg --guardar-json
# Guarda en: data/kyc/resultados/cedula_resultado.json
```

### Ver anotaciones visuales

```bash
python scripts/procesar_cedula.py --imagen cedula.jpg --mostrar-anotaciones --verbose
# Guarda imagen anotada en: data/kyc/resultados/cedula_anotado.jpg
```

### Opciones completas

```
--imagen              Ruta a la imagen JPG/PNG (requerido)
--guardar-json        Guarda resultado en data/kyc/resultados/
--mostrar-anotaciones Guarda imagen con regiones anotadas
--verbose / -v        Muestra OCR raw y detalles del proceso
--modelo-calidad      Ruta a pesos fine-tuned del evaluador (opcional)
```

---

## Uso programatico

```python
from src.kyc_ocr import KYCPipeline

pipeline = KYCPipeline()
resultado = pipeline.procesar("ruta/cedula.jpg")

print(resultado["campos"]["numero"]["valor"])
print(resultado["campos"]["fecha_nacimiento"]["valor"])
print(resultado["campos"]["_meta"]["valido"])
```

### Estructura del JSON de salida

```json
{
  "estado": "procesado",
  "imagen": "ruta/cedula.jpg",
  "calidad": {
    "calidad_score": 0.82,
    "orientacion_grados": 0,
    "aceptable": true,
    "modo": "heuristico"
  },
  "campos": {
    "nombre": {
      "valor": "Juan Perez Garcia",
      "texto_ocr": "JUAN PEREZ GARCIA",
      "confianza": 0.91
    },
    "numero": {
      "valor": "12345678",
      "texto_ocr": "12345678",
      "confianza": 0.95
    },
    "fecha_nacimiento": {
      "valor": "15/03/1985",
      "texto_ocr": "15/03/1985",
      "confianza": 0.88
    },
    "fecha_expedicion": {
      "valor": "20/07/2020",
      "texto_ocr": "20/07/2020",
      "confianza": 0.87
    },
    "_meta": {
      "valido": true,
      "confianza_global": 0.9025,
      "campos_criticos_extraidos": ["numero", "fecha_nacimiento"]
    }
  },
  "tiempo_s": 3.42
}
```

---

## Notas de diseno

### Modo heuristico vs. modelo fine-tuned

El evaluador de calidad (`evaluador_calidad.py`) opera en dos modos:

- **Heuristico** (por defecto): usa varianza de Laplacian para detectar
  desenfoque y analisis de histograma para exposicion. No requiere GPU.
- **ResNet-18 fine-tuned**: si se proporciona `--modelo-calidad`, carga los
  pesos y clasifica en 5 clases (4 orientaciones + rechazo).

### Adaptacion por pais

Las proporciones del template (`segmentador_campos.py`) son configurables.
Para adaptarlo a otro tipo de documento, pasar un dict personalizado a
`segmentar(img, template=mi_template)`.

### Rendimiento en CPU

Todo el pipeline esta disenado para ejecutarse en CPU. La inferencia ResNet-18
en CPU toma ~50ms por imagen; EasyOCR toma ~1-3s por ROI dependiendo del tamano.
