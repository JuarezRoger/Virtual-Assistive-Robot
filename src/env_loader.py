"""
Carga variables desde el archivo .env de la raiz del proyecto.
No sobrescribe variables que ya existan en el entorno.
El .env esta en .gitignore: nunca debe subirse al repositorio.
"""

import os
from pathlib import Path

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def cargar_env(ruta=_ENV_PATH):
    try:
        lineas = Path(ruta).read_text(encoding="utf-8-sig").splitlines()
    except FileNotFoundError:
        return
    for linea in lineas:
        linea = linea.strip()
        if not linea or linea.startswith("#") or "=" not in linea:
            continue
        clave, valor = linea.split("=", 1)
        clave, valor = clave.strip(), valor.strip().strip('"').strip("'")
        if clave and valor and clave not in os.environ:
            os.environ[clave] = valor


cargar_env()
