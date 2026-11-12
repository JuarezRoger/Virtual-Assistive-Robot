"""
ATOM - Modulo de voz
====================
Voz neural en espanol con edge-tts (requiere internet) y respaldo local
con pyttsx3 si falla. Reproduce el MP3 con el MCI nativo de Windows,
sin dependencias extra.

Uso:
    from atom_voz import hablar
    hablar("Hola, soy ATOM")          # bloquea hasta terminar de hablar

La voz se puede cambiar con la variable ATOM_VOZ en el .env
(ej. es-MX-DaliaNeural, es-ES-AlvaroNeural). Lista completa:
    venv\\Scripts\\edge-tts --list-voices
"""

import asyncio
import ctypes
import os
import tempfile
import threading
import uuid

try:
    import env_loader  # noqa: F401  (carga ATOM_VOZ desde .env)
except Exception:
    pass

VOZ = os.environ.get("ATOM_VOZ", "es-MX-JorgeNeural")
VELOCIDAD = os.environ.get("ATOM_VOZ_VELOCIDAD", "+0%")

_lock = threading.Lock()  # evita que dos avisos hablen encima


def _reproducir_mp3(ruta):
    """Reproduce un MP3 de forma bloqueante con winmm (Windows)."""
    winmm = ctypes.windll.winmm
    alias = f"atomvoz{uuid.uuid4().hex[:8]}"
    err = winmm.mciSendStringW(f'open "{ruta}" type mpegvideo alias {alias}', None, 0, 0)
    if err:
        raise RuntimeError(f"MCI open fallo (codigo {err})")
    try:
        winmm.mciSendStringW(f"play {alias} wait", None, 0, 0)
    finally:
        winmm.mciSendStringW(f"close {alias}", None, 0, 0)


def _hablar_edge(texto):
    import edge_tts
    ruta = os.path.join(tempfile.gettempdir(), f"atom_voz_{uuid.uuid4().hex}.mp3")
    try:
        asyncio.run(edge_tts.Communicate(texto, VOZ, rate=VELOCIDAD).save(ruta))
        _reproducir_mp3(ruta)
    finally:
        try:
            os.remove(ruta)
        except OSError:
            pass


def _hablar_local(texto):
    # pyttsx3 solo habla la primera vez por motor; se crea uno nuevo cada vez.
    import pyttsx3
    motor = pyttsx3.init()
    motor.setProperty("rate", 160)
    motor.say(texto)
    motor.runAndWait()
    motor.stop()


def hablar(texto):
    """Lee el texto en voz alta. Retorna 'neural', 'local' o None si fallo."""
    if not texto:
        return None
    with _lock:
        # El servicio de edge-tts falla de forma intermitente: 2 intentos
        for intento in range(2):
            try:
                _hablar_edge(texto)
                return "neural"
            except Exception as e:
                error = e
        print(f"[Voz] edge-tts no disponible ({type(error).__name__}: {error}). Usando voz local.")
        try:
            _hablar_local(texto)
            return "local"
        except Exception as e:
            print(f"[Voz] Error de voz local: {e}")
            return None


if __name__ == "__main__":
    import sys, time
    t = time.time()
    modo = hablar(" ".join(sys.argv[1:]) or "Hola. Soy ATOM, tu asistente de monitoreo cognitivo.")
    print(f"Modo: {modo} | {time.time() - t:.1f}s")
