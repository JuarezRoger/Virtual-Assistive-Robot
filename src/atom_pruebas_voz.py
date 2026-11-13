"""
ATOM - Banco de pruebas de la palabra de activacion
===================================================
Mide con datos reales (tu voz, tu microfono) que tan bien se detecta el
nombre, en lugar de ajustar a ciegas.

  1) Grabar (una vez, ~5 min):   python atom_pruebas_voz.py grabar
     Muestra cada frase, cuenta 3-2-1 y graba 5 s. No hay que tocar nada.
  2) Evaluar (cuantas veces se quiera):   python atom_pruebas_voz.py evaluar

Las grabaciones son tu voz: se guardan en data/pruebas_voz/ (fuera de git).
"""

import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

SR = 16000
DURACION = 5.0
CARPETA = Path(__file__).resolve().parent.parent / "data" / "pruebas_voz"
REPETICIONES_POS = 3
REPETICIONES_NEG = 2

# (variante, frase). Positivos: deben activar. Negativos: NO deben activar.
POSITIVOS = [
    ("atomica", "Oye Atómica"),
    ("atomica", "Atómica, ¿cómo voy?"),
    ("atomica", "Atómica, ¿estás despierta?"),
    ("atom_es", "Oye ATOM  (pronunciado en español: A-tom)"),
    ("atom_es", "ATOM, ¿estás despierto?  (en español)"),
    ("atom_en", "Hey Atom  (pronunciado en inglés: Á-dom)"),
    ("atom_en", "Atom, how am I doing?  (en inglés)"),
    ("atomo", "Átomo, dame un resumen del día"),
]
NEGATIVOS = [
    ("mencion", "El proyecto ATOM mide la fatiga"),
    ("mencion", "La energía atómica es interesante"),
    ("parecida", "Voy a tomar un poco de agua"),
    ("parecida", "Tomo nota de eso"),
    ("parecida", "A todos les mando saludos"),
    ("parecida", "Vaya tontería"),
    ("normal", "Oye, ¿me pasas el cargador?"),
    ("normal", "Hoy tengo mucho trabajo pendiente"),
    ("ruido", "(NO hables: teclea o mueve el mouse)"),
]


def _guardar_wav(ruta, audio):
    with wave.open(str(ruta), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())


def _leer_wav(ruta):
    with wave.open(str(ruta), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768


def grabar():
    import random
    import sounddevice as sd
    CARPETA.mkdir(parents=True, exist_ok=True)
    lista = ([(True, v, f) for v, f in POSITIVOS for _ in range(REPETICIONES_POS)]
             + [(False, v, f) for v, f in NEGATIVOS for _ in range(REPETICIONES_NEG)])
    random.shuffle(lista)   # mezclado: que no se vuelva un sonsonete repetido
    print(f"Se grabarán {len(lista)} frases de {DURACION:.0f} s. Habla normal, "
          "como le hablarías a ATOM. Ctrl+C para cancelar.\n")
    input("Presiona Enter cuando estés listo... ")
    print()
    manifiesto = []
    for n, (positivo, variante, frase) in enumerate(lista, 1):
        print(f"[{n}/{len(lista)}]  >>>  {frase}")
        for c in ("3", "2", "1"):
            print(f"   {c}...", end="", flush=True)
            time.sleep(0.6)
        print("  ¡AHORA!", flush=True)
        audio = sd.rec(int(DURACION * SR), samplerate=SR, channels=1, dtype="float32")
        sd.wait()
        nombre = f"{n:02d}_{'pos' if positivo else 'neg'}_{variante}.wav"
        _guardar_wav(CARPETA / nombre, audio[:, 0])
        manifiesto.append({"archivo": nombre, "positivo": positivo,
                           "variante": variante, "frase": frase})
        print()
    (CARPETA / "manifiesto.json").write_text(
        json.dumps(manifiesto, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Listo: {len(lista)} grabaciones en {CARPETA}")


def evaluar(modelo_nombre="base"):
    from faster_whisper import WhisperModel
    from atom_escucha import PATRON_ACTIVACION, detectar_nombre, transcribir

    manifiesto = json.loads((CARPETA / "manifiesto.json").read_text(encoding="utf-8"))
    modelo = WhisperModel(modelo_nombre, device="cpu", compute_type="int8")
    metodos = {
        "regex_anterior": lambda t: bool(PATRON_ACTIVACION.search(t)),
        "fonetico": lambda t: detectar_nombre(t)[0],
    }
    res = {}   # (metodo, transcripcion) -> lista de (item, detectado)
    print(f"Modelo Whisper: {modelo_nombre}\n")
    for item in manifiesto:
        audio = _leer_wav(CARPETA / item["archivo"])
        t0 = time.time()
        normal = transcribir(modelo, audio)
        seg = time.time() - t0
        con_hot = transcribir(modelo, audio, hotwords="ATOM Atómica")
        marca = "+" if item["positivo"] else "-"
        print(f"{marca} {item['variante']:9s} {seg:4.1f}s | \"{normal}\"  |  hotwords: \"{con_hot}\"")
        for m, f in metodos.items():
            res.setdefault((m, "normal"), []).append((item, f(normal)))
            res.setdefault((m, "hotwords"), []).append((item, f(con_hot)))

    print("\n" + "=" * 70)
    print(f"{'método':15s} {'transcripción':10s} {'detecta':>9s} {'falsas':>8s}   por variante")
    for (m, tr), filas in res.items():
        pos = [d for it, d in filas if it["positivo"]]
        neg = [d for it, d in filas if not it["positivo"]]
        variantes = {}
        for it, d in filas:
            if it["positivo"]:
                variantes.setdefault(it["variante"], []).append(d)
        detalle = "  ".join(f"{v} {sum(x)}/{len(x)}" for v, x in variantes.items())
        print(f"{m:15s} {tr:10s} {sum(pos):3d}/{len(pos):<3d}  {sum(neg):3d}/{len(neg):<3d}   {detalle}")
    falsas = [(it["frase"]) for (m, tr), filas in res.items() if (m, tr) == ("fonetico", "normal")
              for it, d in filas if d and not it["positivo"]]
    if falsas:
        print("\nFalsas activaciones (fonético, normal):", *sorted(set(falsas)), sep="\n  ")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    accion = sys.argv[1] if len(sys.argv) > 1 else ""
    if accion == "grabar":
        grabar()
    elif accion == "evaluar":
        evaluar(sys.argv[2] if len(sys.argv) > 2 else "base")
    else:
        print(__doc__)
