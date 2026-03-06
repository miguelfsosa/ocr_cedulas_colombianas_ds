"""
Modulo 5: Extractor y validador de campos estructurados.

Aplica regex y heuristicas sobre el texto OCR de cada ROI para extraer
campos de cedula colombiana: numero, apellidos, nombres.
"""

import re
import unicodedata


# Etiquetas impresas en la cedula que no son datos del titular
_ETIQUETAS = {
    "REPUBLICA", "COLOMBIA", "IDENTIFICACION", "PERSONAL",
    "CEDULA", "CIUDADANIA", "CIUDADAN", "NUMERO", "NUMENO",
    "NUNINO", "NUVENO", "APELLIDOS", "APLLUDOS", "APLLIDOS",
    "NOMBRES", "NOMBRE", "FIRMA", "DE", "LA", "EL", "NON",
    "ONES", "AQOOO",
}

# Palabras en mayusculas de 3+ letras
_RE_PALABRAS_MAYUS = re.compile(r'[A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ]{2,}')

# Fechas
_RE_FECHA = re.compile(r'\b(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{4})\b')


def _limpiar_texto(texto: str) -> str:
    texto = unicodedata.normalize("NFC", texto)
    texto = re.sub(r'[^\w\s/\-\.áéíóúÁÉÍÓÚñÑ,]', ' ', texto)
    return re.sub(r'\s+', ' ', texto).strip()


def extraer_numero_cedula(texto: str) -> str | None:
    """
    Extrae numero de cedula colombiana (formato X.XXX.XXX.XXX).
    Normaliza coma a punto antes de buscar (OCR a veces confunde separadores).
    """
    texto = _limpiar_texto(texto)
    # Normalizar coma usada como separador de miles a punto
    texto = texto.replace(',', '.')
    # Formato con puntos (mas especifico)
    m = re.search(r'\b\d{1,3}[.]\d{3}[.]\d{3}(?:[.]\d{1,3})?\b', texto)
    if m:
        return m.group(0)
    # Fallback: digitos consecutivos
    m = re.search(r'\b\d{8,10}\b', texto)
    if m:
        return m.group(0)
    return None


def extraer_fecha(texto: str) -> str | None:
    texto = _limpiar_texto(texto)
    m = _RE_FECHA.search(texto)
    if m:
        d, mo, a = m.group(1), m.group(2), m.group(3)
        if 1 <= int(d) <= 31 and 1 <= int(mo) <= 12:
            return f"{int(d):02d}/{int(mo):02d}/{a}"
    return None


def extraer_nombre(texto: str, max_palabras: int = 3) -> str | None:
    """
    Extrae apellidos o nombres del texto OCR.
    - Filtra etiquetas del documento y palabras de menos de 3 letras.
    - Retorna como mucho max_palabras palabras en Title Case.
    - max_palabras=3 cubre casos como "DE LA HOZ" (apellido compuesto).
    """
    texto_norm = unicodedata.normalize("NFC", texto).upper()
    palabras = _RE_PALABRAS_MAYUS.findall(texto_norm)
    validas = [p for p in palabras if p not in _ETIQUETAS and len(p) >= 3]
    if not validas:
        return None
    return " ".join(validas[:max_palabras]).title()


def validar_resultado(campos: dict) -> dict:
    """Agrega metadatos de validacion. Campos criticos: numero y apellidos."""
    campos_criticos = ["numero", "apellidos"]
    extraidos = [
        c for c in campos_criticos
        if isinstance(campos.get(c), dict) and campos[c].get("valor") is not None
    ]
    valido = len(extraidos) == len(campos_criticos)

    confianzas = [
        v["confianza"] for v in campos.values()
        if isinstance(v, dict) and "confianza" in v
    ]
    confianza_global = round(sum(confianzas) / len(confianzas), 4) if confianzas else 0.0

    campos["_meta"] = {
        "valido": valido,
        "confianza_global": confianza_global,
        "campos_criticos_extraidos": extraidos,
    }
    return campos
