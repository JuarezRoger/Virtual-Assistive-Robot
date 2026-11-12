"""
ATOM - Modulo de escucha
========================
Palabra de activacion "ATOM" (u "Oye ATOM") en espanol, 100% local.

Como funciona:
  1. Mide el ruido ambiente del microfono durante 1 s.
  2. Cuando el volumen supera ese ruido, graba el fragmento de voz
     (termina tras ~0.7 s de silencio, maximo 4 s).
  3. Transcribe solo ese fragmento con faster-whisper y busca "ATOM".
En silencio no se ejecuta Whisper, asi que el uso de CPU es minimo.

Uso:
    from atom_escucha import Escucha
    oido = Escucha()
    oido.esperar_activacion()        # bloquea hasta oir "ATOM"
    texto = oido.escuchar_frase()    # transcribe lo siguiente que digas
"""

import re
import time

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

SR = 16000
BLOQUE = 1600                # 100 ms
SILENCIO_FIN = 0.7           # s de silencio que cierran un fragmento
MAX_FRAGMENTO = 4.0          # s maximos para buscar la palabra
FACTOR_RUIDO = 3.0           # voz = RMS > ruido ambiente x este factor
RMS_MINIMO = 0.006
MIN_VOZ = 0.3                # s: fragmentos mas cortos son ruido/clics
MAX_NO_SPEECH = 0.6          # descarta segmentos que Whisper marca como no-voz

# "ATOM" como lo transcribe Whisper en espanol: atom, átomo, atón, a tom...
PATRON_ACTIVACION = re.compile(r"\b(a\s?tomo?|[aá]tomo?s?|at[oó]n)\b", re.IGNORECASE)


class Escucha:
    def __init__(self, modelo_activacion="base", modelo_frases="small"):
        self._w_act = WhisperModel(modelo_activacion, device="cpu", compute_type="int8")
        self._w_frase = (self._w_act if modelo_frases == modelo_activacion
                         else WhisperModel(modelo_frases, device="cpu", compute_type="int8"))
        self.umbral = None

    # ── Microfono ──────────────────────────────────────────────
    def calibrar_ruido(self, segundos=1.0):
        audio = sd.rec(int(SR * segundos), samplerate=SR, channels=1, dtype="float32")
        sd.wait()
        ruido = float(np.sqrt(np.mean(audio ** 2)))
        self.umbral = max(RMS_MINIMO, ruido * FACTOR_RUIDO)
        return ruido

    def _grabar_fragmento(self, stream, max_seg, espera_max=None):
        """Espera voz y graba hasta que haya silencio. None si se agota la espera."""
        inicio = time.time()
        partes, voz_activa, silencio = [], False, 0.0
        while True:
            bloque, _ = stream.read(BLOQUE)
            bloque = bloque.flatten()
            rms = float(np.sqrt(np.mean(bloque ** 2)))
            if not voz_activa:
                if rms > self.umbral:
                    voz_activa = True
                    partes.append(bloque)
                elif espera_max and time.time() - inicio > espera_max:
                    return None
                continue
            partes.append(bloque)
            silencio = silencio + BLOQUE / SR if rms <= self.umbral else 0.0
            if silencio >= SILENCIO_FIN or len(partes) * BLOQUE / SR >= max_seg:
                duracion_voz = len(partes) * BLOQUE / SR - silencio
                if duracion_voz < MIN_VOZ:          # ruido corto: seguir esperando
                    partes, voz_activa, silencio = [], False, 0.0
                    continue
                return np.concatenate(partes)

    def _transcribir(self, modelo, audio):
        # Sin initial_prompt: con audio poco claro Whisper "alucina" el texto
        # de la pista (p. ej. "Oye ATOM.") y provoca falsas activaciones.
        segs, _ = modelo.transcribe(audio, language="es", beam_size=1,
                                    condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segs
                        if s.no_speech_prob < MAX_NO_SPEECH).strip()

    # ── API publica ────────────────────────────────────────────
    def esperar_activacion(self, timeout=None, verbose=False):
        """Bloquea hasta oir 'ATOM'. Retorna el texto oido, o None si timeout."""
        if self.umbral is None:
            self.calibrar_ruido()
        time.sleep(0.4)  # deja apagar el eco de la voz de ATOM en los altavoces
        fin = time.time() + timeout if timeout else None
        with sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=BLOQUE) as st:
            while fin is None or time.time() < fin:
                restante = (fin - time.time()) if fin else None
                audio = self._grabar_fragmento(st, MAX_FRAGMENTO, espera_max=restante)
                if audio is None:
                    return None
                texto = self._transcribir(self._w_act, audio)
                if verbose:
                    print(f"  [oído] \"{texto}\"")
                if PATRON_ACTIVACION.search(texto):
                    return texto
        return None

    def escuchar_frase(self, max_seg=8.0, espera_max=6.0):
        """Graba la siguiente frase y la transcribe con el modelo preciso."""
        if self.umbral is None:
            self.calibrar_ruido()
        time.sleep(0.4)  # deja apagar el eco de la voz de ATOM en los altavoces
        fin = time.time() + espera_max
        with sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=BLOQUE) as st:
            while time.time() < fin:
                audio = self._grabar_fragmento(st, max_seg, espera_max=fin - time.time())
                if audio is None:
                    return ""
                texto = self._transcribir(self._w_frase, audio)
                if texto:
                    return texto
        return ""
