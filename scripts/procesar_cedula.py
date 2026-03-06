#!/usr/bin/env python3
"""
CLI: Procesa una imagen de cedula colombiana y extrae campos estructurados.

Uso:
    python scripts/procesar_cedula.py --imagen ruta/cedula.jpg
    python scripts/procesar_cedula.py --imagen ruta/cedula.jpg --guardar-json --verbose
    python scripts/procesar_cedula.py --imagen ruta/cedula.jpg --mostrar-anotaciones
"""

import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kyc_ocr import KYCPipeline


def _imprimir_resultado(resultado: dict, verbose: bool):
    estado = resultado.get("estado", "desconocido")
    print(f"\n{'='*52}")
    print(f"  RESULTADO KYC  —  Estado: {estado.upper()}")
    print(f"{'='*52}")

    if estado == "rechazado":
        print(f"  Motivo : {resultado.get('motivo', 'N/A')}")
        calidad = resultado.get("calidad", {})
        print(f"  Score  : {calidad.get('calidad_score', 'N/A')}")
        return

    calidad = resultado.get("calidad", {})
    campos = resultado.get("campos", {})
    meta = campos.get("_meta", {})

    if verbose:
        print(f"\n[Calidad de imagen]")
        print(f"  Score      : {calidad.get('calidad_score', 'N/A')}")
        print(f"  Modo       : {calidad.get('modo', 'N/A')}")
        if "varianza_laplacian" in calidad:
            print(f"  Laplacian  : {calidad.get('varianza_laplacian')}")

    print(f"\n[Campos extraidos]")
    for campo in ["numero", "apellidos", "nombres"]:
        info = campos.get(campo, {})
        valor = info.get("valor")
        conf = info.get("confianza", 0.0)
        estado_c = "OK" if valor else "--"
        print(f"  {campo:<12}: {str(valor or '(no detectado)'):<35} [{estado_c}] conf={conf:.2f}")

    if verbose:
        print(f"\n[Texto OCR raw]")
        for campo in ["numero", "apellidos", "nombres"]:
            info = campos.get(campo, {})
            print(f"  {campo:<12}: {info.get('texto_ocr', '')!r}")

    print(f"\n[Validacion]")
    print(f"  Documento valido : {'SI' if meta.get('valido') else 'NO'}")
    print(f"  Confianza global : {meta.get('confianza_global', 0.0):.4f}")
    print(f"  Tiempo proceso   : {resultado.get('tiempo_s', 0):.2f}s")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Extrae campos de una cedula colombiana mediante OCR."
    )
    parser.add_argument("--imagen", required=True,
                        help="Ruta a la imagen JPG/PNG de la cedula.")
    parser.add_argument("--guardar-json", action="store_true",
                        help="Guarda el resultado JSON en data/kyc/resultados/.")
    parser.add_argument("--mostrar-anotaciones", action="store_true",
                        help="Guarda imagen anotada con las regiones detectadas.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Muestra informacion detallada del proceso.")
    parser.add_argument("--modelo-calidad",
                        help="Ruta a pesos fine-tuned del evaluador (opcional).")
    args = parser.parse_args()

    ruta_imagen = Path(args.imagen)
    if not ruta_imagen.exists():
        print(f"[ERROR] Imagen no encontrada: {ruta_imagen}", file=sys.stderr)
        sys.exit(1)

    pipeline = KYCPipeline(ruta_modelo_calidad=args.modelo_calidad)

    base_dir = Path(__file__).parent.parent
    dir_resultados = base_dir / "data" / "kyc" / "resultados"

    ruta_json = None
    ruta_anotacion = None

    if args.guardar_json:
        dir_resultados.mkdir(parents=True, exist_ok=True)
        ruta_json = dir_resultados / (ruta_imagen.stem + "_resultado.json")

    if args.mostrar_anotaciones:
        dir_resultados.mkdir(parents=True, exist_ok=True)
        ruta_anotacion = dir_resultados / (ruta_imagen.stem + "_anotado.jpg")

    if args.verbose:
        print(f"Procesando: {ruta_imagen}")

    resultado = pipeline.procesar(ruta_imagen, guardar_anotacion=ruta_anotacion)
    _imprimir_resultado(resultado, verbose=args.verbose)

    if ruta_json:
        json_str = json.dumps(resultado, ensure_ascii=False, indent=2)
        ruta_json.write_text(json_str, encoding="utf-8")
        print(f"JSON guardado en : {ruta_json}")

    if ruta_anotacion and ruta_anotacion.exists():
        print(f"Anotacion guardada: {ruta_anotacion}")


if __name__ == "__main__":
    main()
