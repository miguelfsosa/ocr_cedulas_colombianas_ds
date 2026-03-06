# Pipeline OCR para Cedula Colombiana: Explicacion Tecnica Detallada

**Proyecto:** Deep Learning Banking - Modulo KYC
**Audiencia:** Cientificos de datos e ingenieros de ML
**Objetivo:** Explicar paso a paso como funciona el sistema de extraccion automatica
de campos de la Cedula de Ciudadania colombiana, desde la foto cruda hasta el JSON.

---

## Vision General

El sistema recibe una fotografia de una cedula tomada en condiciones reales
(iluminacion variable, perspectiva, fondo) y retorna un JSON con los campos del
documento: numero de cedula, apellidos y nombres.

El pipeline tiene **5 modulos secuenciales**, cada uno con responsabilidad delimitada:

```
Foto JPG/PNG
     |
     v
[Modulo 1] Preprocesamiento         -> imagen normalizada y rectificada
     |
     v
[Modulo 2] Evaluador de Calidad     -> score + decision de aceptar/rechazar
     |
     v
[Modulo 3] Segmentador de Campos    -> recortes (ROIs) por campo
     |
     v
[Modulo 4] OCR (EasyOCR)            -> texto crudo + confianza por region
     |
     v
[Modulo 5] Extractor y Validador    -> campos limpios + metadatos
     |
     v
JSON estructurado
```

---

## Modulo 1: Preprocesamiento

**Archivo:** `src/kyc_ocr/preprocesamiento.py`
**Proposito:** Estandarizar la imagen de entrada para maximizar la legibilidad
del texto antes del OCR. Aplica 5 transformaciones en cadena.

---

### 1.1 Redimensionado

Si la imagen tiene mas de 1280 pixeles de ancho se reduce proporcionalmente con
interpolacion **INTER_AREA**. Esta interpolacion promedia los pixeles del area
original en lugar de interpolar valores nuevos, preservando la nitidez del texto.

1280px es suficiente para que EasyOCR lea texto de cedula y evita tiempos
excesivos con fotos de smartphones modernos (4000-8000px de ancho tipicamente).

---

### 1.2 Deteccion del documento

**Objetivo:** identificar los 4 vertices de la cedula dentro de la foto para
poder rectificarla. Proceso de 6 pasos:

**Paso 1 — Conversion a escala de grises:** canal unico para el calculo de bordes.
La informacion de color no aporta en esta etapa.

**Paso 2 — Suavizado Gaussiano** (kernel 5x5): reduce ruido de alta frecuencia
que generaria falsos bordes en Canny. El kernel Gaussiano promedia cada pixel con
sus vecinos ponderando inversamente a la distancia.

**Paso 3 — Detector de bordes Canny** (umbrales 75 y 200):

Internamente calcula el gradiente de intensidad con el operador de Sobel en X e Y,
luego aplica supresion de no-maximos (conserva solo los pixeles que son maximos
locales en la direccion del gradiente), y finalmente umbraliza con histeresis:
un borde debil (entre 75 y 200) se conserva solo si esta conectado a un borde
fuerte (>200). Esto elimina bordes espurios y conecta contornos fragmentados.

**Paso 4 — Busqueda de contornos:** se recuperan todos los contornos y se
seleccionan los 5 de mayor area (candidatos mas probables a ser el documento).

**Paso 5 — Aproximacion poligonal** con el algoritmo de Ramer-Douglas-Peucker:
simplifica cada contorno eliminando puntos redundantes con una tolerancia del 2%
del perimetro. Si el poligono resultante tiene exactamente 4 vertices, se asume
que es el borde del documento.

**Paso 6 — Ordenamiento de esquinas:** los 4 puntos se ordenan como
[arriba-izq, arriba-der, abajo-der, abajo-izq] usando una propiedad geometrica:
la esquina superior-izquierda tiene la menor suma x+y; la inferior-derecha la
mayor suma; las otras dos se distinguen por la diferencia x-y.

Si ningun contorno produce exactamente 4 vertices (caso comun con fondos
texturados como madera o tela que generan muchos contornos competidores),
retorna `None` y se usa la imagen completa sin recortar (fallback).

---

### 1.3 Correccion de perspectiva

Cuando la cedula fue fotografiada en angulo, los caracteres aparecen
distorsionados (trapezoidales en lugar de rectangulares). Esta funcion aplica
una **transformacion homografica** para obtener una vista frontal perfecta.

**Matematicamente:** dado un conjunto de 4 puntos fuente (esquinas detectadas)
y 4 puntos destino (vertices de un rectangulo plano), se calcula la **matriz de
perspectiva H** de 3x3 que mapea cada punto del plano de la cedula al plano de
la imagen rectificada. H se obtiene resolviendo un sistema de 8 ecuaciones
lineales (4 pares de puntos, 2 ecuaciones por par) con 8 incognitas (H tiene 9
elementos con un grado de libertad de escala).

```python
M = getPerspectiveTransform(esquinas_origen, esquinas_destino)
img_rect = warpPerspective(img, M, (ancho_max, alto_max))
```

El tamano del rectangulo destino se calcula con la distancia euclidiana entre
pares de esquinas paralelas, tomando el maximo para no perder resolucion.

---

### 1.4 Mejora de contraste: CLAHE

**CLAHE** (Contrast Limited Adaptive Histogram Equalization) divide la imagen en
tiles de 8x8 pixeles y equaliza el histograma de cada tile de forma independiente,
pero con un **limite de amplificacion** (clipLimit=2.0): si un bin del histograma
excede ese limite, los pixeles sobrantes se redistribuyen uniformemente en los
demas bins antes de ecualizar. Produce mejora de contraste local sin amplificar
ruido, a diferencia de la ecualizacion global de histograma.

**Por que el canal L del espacio LAB:** el espacio de color CIELAB separa la
luminancia (L) de la informacion de color (canales A y B). CLAHE se aplica solo
sobre L para mejorar el contraste sin alterar los colores originales — aplicarlo
en BGR cambiaria la tonalidad de la imagen.

---

### 1.5 Reduccion de ruido: Non-Local Means (NLM)

A diferencia de filtros locales (Gaussiano, mediana) que usan solo los pixeles
vecinos inmediatos, **NLM** busca en toda la imagen parches de 7x7 pixeles
similares al parche que rodea cada pixel, y los promedia ponderadamente. El peso
de cada parche candidato es proporcional a su similitud fotometrica:

```
w(i,j) = exp( -||P(i) - P(j)||^2 / h^2 )
```

donde P(i) y P(j) son los parches comparados y h=10 es el parametro de filtrado.

Resultado: elimina el granulado digital sin borrar los bordes de los caracteres,
porque aunque un pixel en el borde de una letra tenga vecinos de intensidades
distintas, en otra parte de la imagen hay un patron similar (otro tramo de la
misma letra) que ayuda a estimar su valor sin difuminar el borde.

---

## Modulo 2: Evaluador de Calidad

**Archivo:** `src/kyc_ocr/evaluador_calidad.py`
**Proposito:** Determinar si la imagen tiene suficiente calidad para OCR confiable.
Opera en dos modos segun si hay un modelo fine-tuned disponible.

---

### 2.1 Modo Heuristico (activo por defecto)

Dos metricas calculadas directamente sin red neuronal:

**Metrica 1 — Varianza del Laplaciano (nitidez):**

El operador Laplaciano calcula la segunda derivada de la intensidad. En 2D discreto:

```
nabla^2 f(x,y) = f(x+1,y) + f(x-1,y) + f(x,y+1) + f(x,y-1) - 4*f(x,y)
```

Imagen nitida: bordes con transiciones abruptas → segunda derivada alta →
varianza del mapa Laplaciano alta.

Imagen borrosa: transiciones suaves → segunda derivada baja en todas partes →
varianza baja.

Se normaliza: `blur_norm = min(varianza / 300, 1.0)`, donde 300 es un valor de
referencia empirico para imagenes de cedulas (imagenes nitidas tipicamente
producen varianzas entre 80 y 500).

**Metrica 2 — Score de exposicion:**

```
hist = histograma de 256 bins de la imagen en grises
extremos = porcentaje de pixeles con intensidad < 20 o > 235
score_exposicion = 1.0 - extremos
```

Penaliza imagenes subexpuestas (texto negro sobre fondo negro) o sobreexpuestas
(texto blanco sobre fondo blanco), donde el OCR no puede distinguir caracteres.

**Score combinado:**

```
calidad_score = 0.6 x blur_norm + 0.4 x score_exposicion
```

Se prioriza nitidez (60%) sobre exposicion (40%) porque el desenfoque degrada
mas el OCR que una iluminacion imperfecta.

**Decision del pipeline:** si `calidad_score < umbral_calidad` (default 0.25),
la imagen se rechaza retornando `estado: "rechazado"` sin continuar el proceso.
El umbral es ajustable en tiempo de ejecucion sin reentrenar ningun modelo.

---

### 2.2 Modo ResNet-18 (con modelo fine-tuned)

Cuando se proporciona un checkpoint de pesos, se usa una red neuronal convolucional.

**Arquitectura ResNet-18:**

ResNet-18 tiene 18 capas con ~11 millones de parametros, preentrenada en
ImageNet (1.2M imagenes, 1000 clases). La innovacion clave son los **bloques
residuales**: en lugar de aprender la transformacion H(x), la red aprende el
residuo F(x) = H(x) - x, de modo que H(x) = F(x) + x. Esto se implementa con
una **skip connection** que suma la entrada del bloque a la salida de las capas
convolucionales.

Esta formulacion permite el flujo directo del gradiente hacia capas tempranas
durante backpropagation, resolviendo el problema de desvanecimiento del gradiente
en redes profundas.

```
Imagen 224x224x3 (RGB, normalizado con media y std de ImageNet)
     |
     v
Conv 7x7 stride 2 → MaxPool 3x3 → 64 canales (56x56)
4 grupos de bloques residuales: 64 → 128 → 256 → 512 canales
     |
     v
Average Pooling Global → vector 512-dimensional
     |
     v
Cabecera reemplazada (Transfer Learning):
  Linear(512→256) → ReLU → Dropout(0.3) → Linear(256→5)
     |
     v
Softmax → probabilidades sobre 5 clases:
  Clase 0: orientacion correcta (0 grados)
  Clase 1: imagen rotada 90 grados
  Clase 2: imagen invertida (180 grados)
  Clase 3: imagen rotada 270 grados
  Clase 4: imagen rechazada (borrosa, ilegible, no es cedula)
```

**Por que Transfer Learning:** las primeras capas de ResNet-18 preentrenada
aprenden detectores de bordes, texturas y formas aplicables a cualquier imagen.
Al congelar esas capas y reentrenar solo la cabecera, se necesitan ~500 imagenes
por clase en lugar de millones.

**Normalizacion de entrada:** se aplica la normalizacion de ImageNet
(`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`), preservando
la compatibilidad con las representaciones internas del backbone.

---

## Modulo 3: Segmentador de Campos

**Archivo:** `src/kyc_ocr/segmentador_campos.py`
**Proposito:** Dividir la imagen en Regiones de Interes (ROIs) por campo.

---

### Template para cedula colombiana

```python
TEMPLATE_CEDULA_COLOMBIANA = {
    # campo:     (y_inicio, y_fin, x_inicio, x_fin)  en [0, 1]
    "foto":      (0.13, 0.87, 0.60, 0.97),
    "numero":    (0.18, 0.37, 0.02, 0.60),
    "apellidos": (0.35, 0.55, 0.02, 0.60),
    "nombres":   (0.53, 0.72, 0.02, 0.60),
    "firma":     (0.70, 0.90, 0.02, 0.55),
}
```

Todas las coordenadas son **fracciones del alto y ancho de la imagen**, haciendo
el template invariante a la resolucion. Por ejemplo, la region de apellidos
abarca el 35-55% del alto y el 2-60% del ancho.

**Por que template-based:** la deteccion automatica de campos (modelos de
segmentacion semantica o CRAFT) requiere datos de entrenamiento anotados campo
por campo. El template funciona bien para documentos con layout fijo y estandarizado,
que es exactamente el caso de la cedula colombiana (mismo formato desde 2000).

Recorte de cada ROI — slicing matricial directo sobre el array NumPy:

```python
roi = img[int(y0*alto):int(y1*alto), int(x0*ancho):int(x1*ancho)].copy()
```

El `.copy()` garantiza un array independiente en memoria.

**Ejemplo concreto para imagen de 960x1280px:**

| Campo | Proporciones | Pixeles resultantes |
|-------|-------------|---------------------|
| numero | y: 18-37%, x: 2-60% | 182 x 742 px |
| apellidos | y: 35-55%, x: 2-60% | 192 x 742 px |
| nombres | y: 53-72%, x: 2-60% | 182 x 742 px |

---

## Modulo 4: OCR con EasyOCR

**Archivo:** `src/kyc_ocr/pipeline.py`
**Proposito:** Convertir cada ROI en texto plano con una puntuacion de confianza.

---

### Arquitectura interna de EasyOCR

EasyOCR es un sistema de dos etapas independientes:

**Etapa 1 — Deteccion de texto (CRAFT):**

CRAFT (Character Region Awareness for Text Detection) es una red basada en
VGG-16 que produce dos mapas de calor sobre la imagen:

- **Mapa de regiones de caracteres:** alta activacion en el centro geometrico
  de cada caracter individual
- **Mapa de afinidad entre caracteres:** alta activacion en el espacio entre
  pares de caracteres consecutivos de una misma palabra

Con la combinacion de estos mapas se generan bounding boxes por umbralizado y
deteccion de componentes conectadas.

CRAFT detecta texto en cualquier orientacion y curvatura, siendo robusto a
condiciones reales. Supera a los metodos clasicos (morfologia, proyeccion de
histograma) que asumen texto horizontal y perfectamente alineado.

**Etapa 2 — Reconocimiento de texto (CRNN + CTC):**

Para cada bounding box detectado se aplica una CRNN:

```
Imagen del bbox (altura fija h=32px, ancho variable)
     |
     v
Capas convolucionales tipo VGG
→ un vector de features por cada columna de la imagen
     |
     v
2 capas LSTM bidireccionales
→ modela dependencias entre caracteres (contexto izq. y der.)
     |
     v
Linear(512 → |vocabulario|) por paso de tiempo
     |
     v
Decodificacion CTC → string de texto
```

**CTC (Connectionist Temporal Classification):** agrega un caracter "blank"
al vocabulario y permite que la red produzca multiples frames por caracter sin
necesitar alineacion explicita. La decodificacion colapsa repeticiones y remueve
blanks para obtener el texto final. Esto es clave porque evita la necesidad de
anotar la posicion exacta de cada caracter durante el entrenamiento.

Las **LSTMs bidireccionales** capturan contexto en ambas direcciones, resolviendo
ambiguedades como "rn" vs "m" usando los caracteres adyacentes.

---

### Configuracion y agregacion de resultados

```python
reader = easyocr.Reader(["es", "en"], verbose=False)
resultados = reader.readtext(roi, detail=1)
# Cada elemento: (bbox, texto_detectado, confianza)

texto_unido = " ".join([r[1] for r in resultados])
confianza_promedio = mean([r[2] for r in resultados])
```

Cuando el OCR detecta multiples bloques en la misma ROI (por ejemplo, la etiqueta
impresa "NUMERO" y el numero "1.094.957.337"), se concatenan y sus confianzas se
promedian. La limpieza de etiquetas se delega al Modulo 5.

---

## Modulo 5: Extractor y Validador de Campos

**Archivo:** `src/kyc_ocr/extractor_campos.py`
**Proposito:** Convertir el texto OCR en campos estructurados, filtrando ruido,
artefactos y etiquetas impresas del documento.

---

### 5.1 Extraccion del numero de cedula

```python
texto = texto.replace(",", ".")   # normalizar separador de miles
m = re.search(r"\d{1,3}[.]\d{3}[.]\d{3}(?:[.]\d{1,3})?", texto)
```

**Diseno del regex para formato X.XXX.XXX.XXX:**

| Fragmento regex | Significado |
|----------------|-------------|
| `\d{1,3}` | Millones: 1-3 digitos (ej. "1" en 1.094.957.337) |
| `[.]\d{3}` | Punto separador + exactamente 3 digitos de miles |
| `[.]\d{3}` | Segundo separador + 3 digitos de unidades de mil |
| `(?:[.]\d{1,3})?` | Tercer separador + digito final (opcional) |

La normalizacion **coma → punto** es critica: el OCR frecuentemente confunde el
punto tipografico del separador de miles con una coma (diferencia de 1-2 pixeles
en tipografias pequenas). Sin esta normalizacion, "1.037.641,402" no matchearia
el regex y se perderia el ultimo bloque del numero.

Fallback: si el formato con puntos falla, se busca una secuencia de 8-10 digitos
consecutivos (cedulas antiguas o lecturas OCR sin separadores).

---

### 5.2 Extraccion de apellidos y nombres

```python
palabras = re.findall(r"[A-Z][A-Z]{2,}", texto.upper())
validas = [p for p in palabras if p not in ETIQUETAS and len(p) >= 3]
return " ".join(validas[:3]).title()
```

**Logica paso a paso:**

1. **Normalizacion a mayusculas:** unifica la salida del OCR que mezcla casos
   ("Jara arteaga Aqooo" → "JARA ARTEAGA AQOOO").

2. **Regex de palabras >=3 letras en mayusculas:** excluye numeros, simbolos y
   abreviaciones de un solo caracter.

3. **Lista negra de etiquetas del documento:** la cedula tiene texto fijo impreso
   que el OCR captura por solapamiento de los bordes de la ROI con las etiquetas
   tipograficas:

   ```python
   ETIQUETAS = {
       "REPUBLICA", "COLOMBIA", "IDENTIFICACION", "PERSONAL",
       "CEDULA", "CIUDADANIA", "NUMERO", "APELLIDOS", "NOMBRES",
       "FIRMA", "NON", "AQOOO", "ONES", "APLLUDOS", ...
   }
   ```

   Algunas entradas como "AQOOO" o "NON" son artefactos OCR recurrentes
   identificados al correr el pipeline sobre imagenes de muestra — se agregan
   conforme se observan en produccion.

4. **Limite de 3 palabras:** cubre apellidos compuestos como "DE LA HOZ" sin
   incluir texto espurio adicional mas alla.

5. **Title Case:** normaliza "JARA ARTEAGA" → "Jara Arteaga".

---

### 5.3 Validacion del resultado

Los **campos criticos** para la cedula colombiana son `numero` y `apellidos`.
Un documento se marca como valido si ambos fueron extraidos con valor no nulo.
Los nombres son deseables pero no criticos (el ROI de nombres puede solapar con
la firma produciendo OCR degradado en algunos casos).

La **confianza global** es el promedio aritmetico de las confianzas de EasyOCR
para cada campo, usada como indicador de certeza de la extraccion completa.

---

## Flujo Completo: Ejemplo Real

Para `cedula_1.jpeg` — Maria Alejandra Jara Arteaga, 1.094.957.337:

```
Entrada: foto JPG 3024x4032px, cedula sobre mesa de madera

Modulo 1 - Preprocesamiento:
  Redimensionar: 3024x4032 → 960x1280px
  Detectar documento: fondo de madera genera contornos espurios →
    ningun cuadrilatero limpio de 4 vertices → fallback imagen completa
  CLAHE canal L: mejora contraste local
  NLM Denoising h=10: reduce granulado digital

Modulo 2 - Calidad heuristica:
  Varianza Laplaciana: 16.52
  blur_norm: min(16.52/300, 1.0) = 0.055
  Score exposicion: 0.688
  calidad_score: 0.6*0.055 + 0.4*0.688 = 0.033 + 0.275 = 0.308
  Umbral 0.25: 0.308 > 0.25 → ACEPTAR

  Nota: blur_norm es bajo no porque la imagen sea borrosa, sino porque
  al no recortar el documento del fondo, el Laplaciano promedia la
  textura de la madera (suave) con el texto (nitido).

Modulo 3 - Segmentacion (imagen 960x1280px):
  "numero":    img[173:355, 26:768]  → 182x742px
  "apellidos": img[336:528, 26:768]  → 192x742px
  "nombres":   img[509:691, 26:768]  → 182x742px

Modulo 4 - OCR por ROI:
  "numero":    "( CEDULA DE Ciudadan nunino 1.094.957.337"  conf=0.29
  "apellidos": "Jara arteaga Aqooo"                        conf=0.34
  "nombres":   "Maria Alejandra Non"                       conf=0.28

Modulo 5 - Extraccion:
  numero:    regex encuentra "1.094.957.337" directamente → OK
  apellidos: ["JARA","ARTEAGA","AQOOO"] → filtrar "AQOOO" → "Jara Arteaga"
  nombres:   ["MARIA","ALEJANDRA","NON"] → filtrar "NON" → "Maria Alejandra"

JSON de salida:
{
  "estado": "procesado",
  "calidad": {"calidad_score": 0.4326, "modo": "heuristico"},
  "campos": {
    "numero":    {"valor": "1.094.957.337",   "confianza": 0.29},
    "apellidos": {"valor": "Jara Arteaga",    "confianza": 0.34},
    "nombres":   {"valor": "Maria Alejandra", "confianza": 0.28},
    "_meta": {"valido": true, "confianza_global": 0.3032}
  },
  "tiempo_s": 11.16
}
```

---

## Limitaciones y Lineas de Mejora

### Limitaciones actuales

| Componente | Limitacion | Impacto |
|------------|-----------|---------|
| Deteccion de documento | Falla con fondos texturados | Template sobre imagen completa → proporciones desviadas |
| Template fijo | Solo formato estandar moderno | Cedulas antiguas o danadas pueden desalinearse |
| OCR confianza baja | Fondos complejos + pattern globo terraqueo | Campos correctos pero score bajo (~0.29) |
| Evaluador heuristico | No detecta orientacion | Cedulas rotadas se procesan igual |
| Lista negra | Hardcodeada manualmente | Artefactos OCR nuevos no previstos pueden pasar |

### Mejoras tecnicas de mayor impacto

**1. Deteccion de documento por color (corto plazo):**

La cedula colombiana es blanca/lila sobre cualquier fondo. Segmentar en espacio
HSV para detectar el documento sin depender de Canny:

```python
hsv = cvtColor(img, BGR2HSV)
mascara = inRange(hsv, lower=[0,0,180], upper=[180,60,255])
contorno = max(findContours(mascara), key=contourArea)
```

**2. Fine-tuning del ResNet-18 (mediano plazo):**

Con ~500 imagenes por clase (bien encuadrada / borrosa / rotada / ilegible)
se puede entrenar el evaluador CNN. El modelo aprende automaticamente que rasgos
visuales indican calidad en el contexto especifico de cedulas colombianas,
superando las metricas heuristicas generales.

**3. Binarizacion adaptativa por ROI con Sauvola (corto plazo):**

Aplicar binarizacion local directamente sobre cada ROI antes del OCR. Sauvola
calcula un umbral por pixel usando estadisticas de vecindad:

```
T(x,y) = mean(x,y) * [1 + k * (std(x,y)/R - 1)]
k ~ 0.2, R = 128
```

Muy efectivo para texto sobre fondos con patron como el globo terraqueo
impreso en la cedula.

**4. Validacion por digito verificador (largo plazo):**

Implementar la formula de verificacion del numero de cedula para detectar
errores de OCR en el numero sin consultar bases de datos externas.

---

## Tecnologias y Dependencias

| Libreria | Version minima | Rol en el pipeline |
|----------|---------------|-------------------|
| OpenCV | 4.8 | Canny, CLAHE, NLM, warpPerspective, contornos |
| EasyOCR | 1.7 | CRAFT (deteccion) + CRNN/CTC (reconocimiento) |
| PyTorch | 2.0 | Backbone ResNet-18 del evaluador de calidad |
| torchvision | 0.15 | Modelos preentrenados y transforms de normalizacion |
| NumPy | 1.24 | Operaciones matriciales sobre arrays de imagen |

**Tiempos en CPU:** primera inferencia 8-12s (carga de modelos EasyOCR en RAM),
siguientes 2-4s por imagen. Con GPU: menos de 1s por imagen.

---

*Proyecto Deep Learning Banking — modulo KYC/OCR*
