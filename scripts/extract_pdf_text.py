"""
Script de extracción de texto del PDF del Capítulo 2
usando PyMuPDF (renderizado) + EasyOCR (reconocimiento de texto).

Uso:
    python scripts/extract_pdf_text.py

Output:
    lecturas/chapter2_extracted.txt  - Texto crudo extraído por página
"""

import sys
import time
from pathlib import Path

# Forzar UTF-8 en stdout/stderr para evitar errores en Windows con caracteres Unicode
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Rutas del proyecto
PROYECTO_DIR = Path(__file__).parent.parent
PDF_PATH = PROYECTO_DIR / "lecturas" / "deep_learning_banking_chapter_2.pdf"
OUTPUT_PATH = PROYECTO_DIR / "lecturas" / "chapter2_extracted.txt"

# Parámetros de extracción
DPI = 150           # Balance entre velocidad y calidad OCR
LOTE_SIZE = 10      # Guardar progreso cada N páginas


def extraer_texto_pdf():
    """Renderiza cada página del PDF y aplica OCR para extraer el texto."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("ERROR: PyMuPDF no instalado. Ejecutar: pip install pymupdf")
        sys.exit(1)

    try:
        import easyocr
    except ImportError:
        print("ERROR: EasyOCR no instalado. Ejecutar: pip install easyocr")
        sys.exit(1)

    import numpy as np

    if not PDF_PATH.exists():
        print(f"ERROR: No se encontró el PDF en {PDF_PATH}")
        sys.exit(1)

    print(f"Abriendo PDF: {PDF_PATH}")
    doc = fitz.open(str(PDF_PATH))
    total_paginas = len(doc)
    print(f"Total de páginas: {total_paginas}")

    print("Inicializando EasyOCR (puede descargar modelos la primera vez ~500MB)...")
    lector = easyocr.Reader(['en'], gpu=False, verbose=False)
    print("EasyOCR listo.\n")

    texto_completo = []
    tiempo_inicio = time.time()

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as archivo_salida:
        archivo_salida.write(f"# Texto extraído de: {PDF_PATH.name}\n")
        archivo_salida.write(f"# Total páginas: {total_paginas}\n")
        archivo_salida.write(f"# DPI de renderizado: {DPI}\n")
        archivo_salida.write("=" * 80 + "\n\n")

        for num_pagina in range(total_paginas):
            tiempo_pagina = time.time()

            # Renderizar página como imagen
            pagina = doc[num_pagina]
            mat = fitz.Matrix(DPI / 72, DPI / 72)  # Factor de escala según DPI
            pixmap = pagina.get_pixmap(matrix=mat)

            # Convertir a array numpy para EasyOCR
            img_array = np.frombuffer(pixmap.samples, dtype=np.uint8)
            img_array = img_array.reshape(pixmap.height, pixmap.width, pixmap.n)

            # Si el pixmap tiene canal alpha (4 canales), convertir a RGB
            if pixmap.n == 4:
                img_array = img_array[:, :, :3]

            # Aplicar OCR
            resultados = lector.readtext(img_array, detail=0, paragraph=True)
            texto_pagina = "\n".join(resultados)

            # Escribir en archivo con marcador de página
            encabezado = f"\n\n{'='*80}\n## PÁGINA {num_pagina + 1}\n{'='*80}\n\n"
            archivo_salida.write(encabezado)
            archivo_salida.write(texto_pagina)
            archivo_salida.flush()

            # Progreso
            tiempo_transcurrido = time.time() - tiempo_inicio
            tiempo_por_pagina = tiempo_transcurrido / (num_pagina + 1)
            paginas_restantes = total_paginas - num_pagina - 1
            tiempo_estimado = paginas_restantes * tiempo_por_pagina

            print(
                f"Página {num_pagina + 1:3d}/{total_paginas} | "
                f"{time.time() - tiempo_pagina:.1f}s | "
                f"Restante: {tiempo_estimado/60:.1f} min | "
                f"Palabras: {len(texto_pagina.split())}"
            )

    doc.close()
    tiempo_total = time.time() - tiempo_inicio
    print(f"\nExtracción completada en {tiempo_total/60:.1f} minutos")
    print(f"Texto guardado en: {OUTPUT_PATH}")

    # Estadísticas finales
    with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
        contenido = f.read()
    print(f"Total caracteres: {len(contenido):,}")
    print(f"Total palabras: {len(contenido.split()):,}")


if __name__ == "__main__":
    extraer_texto_pdf()
