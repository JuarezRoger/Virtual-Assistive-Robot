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

import queue
import re
import time

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

from atom_voz import atom_esta_hablando

SR = 16000
BLOQUE = 1600                # 100 ms
SILENCIO_FIN = 0.7           # s de silencio que cierran un fragmento
MAX_FRAGMENTO = 8.0          # s maximos (cabe "Oye ATOM" + una pregunta)
FACTOR_RUIDO = 3.0           # voz = RMS > ruido ambiente x este factor
RMS_MINIMO = 0.006
MIN_VOZ = 0.3                # s: fragmentos mas cortos son ruido/clics
MAX_NO_SPEECH = 0.6          # descarta segmentos que Whisper marca como no-voz
MIN_LOGPROB = -1.2           # confianza minima del segmento. -1.0 (estandar) descartaba
                             # frases correctas: "Atom" no es palabra espanola y Whisper
                             # la escribe con poca confianza (medido en atom_pruebas_voz)
MAX_COMPRESION = 2.4         # texto repetitivo = alucinacion (umbral estandar Whisper)

# Regex anterior (se conserva solo para comparar en atom_pruebas_voz.py)
PATRON_ACTIVACION = re.compile(r"\b(a\s?tomo?|[aá]tomo?s?|at[oó]n)\b", re.IGNORECASE)

# ── Deteccion del nombre por sonido ─────────────────────────────
# Whisper escribe "ATOM" de muchas formas ("atón", "a tomo", "Vaya ton").
# En vez de perseguir cada variante con regex, se normaliza el texto a una
# forma fonetica simple y se compara con los nombres validos.
# Atómica, ATOM, Átomo normalizados + como los oye Whisper en grabaciones reales:
# "Oye, Tomica" (la a se pega al oye), "a tome"
NOMBRES = ("atomica", "tomica", "aton", "atome", "atomo")
SALUDOS = {"oye", "ey", "ei", "ola", "oiga", "eh", "ok", "oc", "bueno", "ai"}
SIMILITUD_NOMBRE = 0.85                   # nombres cortos: casi exactos


def _normalizar(palabra):
    import unicodedata
    p = unicodedata.normalize("NFD", palabra.lower())
    p = "".join(c for c in p if unicodedata.category(c) != "Mn")   # sin acentos
    p = p.replace("h", "").replace("v", "b").replace("z", "s").replace("k", "c")
    return p[:-1] + "n" if p.endswith("m") else p                  # atom -> aton


def detectar_nombre(texto):
    """Busca el nombre AL INICIO de la frase (tras un saludo opcional).

    'Oye ATOM, ¿cómo voy?'    -> (True, '¿cómo voy?')
    'El proyecto ATOM mide…'  -> (False, '')   hablar *de* ATOM no lo activa
    'Vaya ton, ¿cómo voy?'    -> (True, '¿cómo voy?')   ('oye a-tom' mal oído)
    """
    from difflib import SequenceMatcher
    tokens = list(re.finditer(r"[^\W\d_]+", texto))
    norm = [_normalizar(t.group()) for t in tokens]
    i = 0
    while i < len(norm) and i < 2 and norm[i] in SALUDOS:
        i += 1
    # Une hasta 3 palabras desde el inicio: "a tomo" -> "atomo"
    for k in range(1, 4):
        if i + k > len(norm):
            break
        bloque = "".join(norm[i:i + k])
        limites = {sum(len(n) for n in norm[i:i + j]) for j in range(1, k + 1)}
        for inicio in range(0, min(4, len(bloque))):
            # Solo puede empezar dentro de la primera palabra ("baya|ton")
            if inicio > 0 and k == 1:
                continue
            for nombre in NOMBRES:
                if bloque[inicio] != nombre[0]:
                    continue
                for largo in (len(nombre) - 1, len(nombre), len(nombre) + 1):
                    fin = inicio + largo
                    if fin not in limites:      # debe terminar donde termina una palabra
                        continue
                    if SequenceMatcher(None, bloque[inicio:fin], nombre).ratio() >= SIMILITUD_NOMBRE:
                        palabras = sorted(limites).index(fin) + 1
                        resto = texto[tokens[i + palabras - 1].end():]
                        return True, re.sub(r"^[\s,.;:!¡?¿-]+", "", resto).strip()
    return False, ""


def transcribir(modelo, audio, debug=False, hotwords=None):
    """Transcribe y descarta segmentos dudosos (alucinaciones de Whisper)."""
    # Sin initial_prompt: con audio poco claro Whisper "alucina" el texto
    # de la pista (p. ej. "Oye ATOM.") y provoca falsas activaciones.
    segs, _ = modelo.transcribe(audio, language="es", beam_size=1,
                                condition_on_previous_text=False, hotwords=hotwords)
    validos = []
    for seg in segs:
        if (seg.no_speech_prob >= MAX_NO_SPEECH or seg.avg_logprob < MIN_LOGPROB
                or seg.compression_ratio > MAX_COMPRESION):
            if debug:
                print(f"  [descartado] '{seg.text.strip()}' (no_voz {seg.no_speech_prob:.2f}, "
                      f"logprob {seg.avg_logprob:.2f}, compresion {seg.compression_ratio:.1f})")
            continue
        validos.append(seg.text.strip())
    return " ".join(validos).strip()


class Escucha:
    def __init__(self, modelo_activacion="base", modelo_frases="small"):
        self._w_act = WhisperModel(modelo_activacion, device="cpu", compute_type="int8")
        self._w_frase = (self._w_act if modelo_frases == modelo_activacion
                         else WhisperModel(modelo_frases, device="cpu", compute_type="int8"))
        self.umbral = None
        self.debug = False
        # El microfono graba continuamente a una cola desde el hilo de audio,
        # asi no se pierden muestras mientras Whisper transcribe (con la CPU
        # ocupada por el sensor visual el audio llegaba cortado).
        self._cola = queue.Queue()
        self.desbordes = 0
        self._stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                                      blocksize=BLOQUE, callback=self._al_recibir)
        self._stream.start()
        self._d_t, self._d_max, self._d_bloq, self._d_n = time.time(), 0.0, 0, 0

    def _diag(self, rms, bloqueado):
        """Cada 10 s imprime el volumen maximo oido (para depurar el microfono)."""
        self._d_max = max(self._d_max, rms)
        self._d_bloq += bloqueado
        self._d_n += 1
        if time.time() - self._d_t >= 10:
            print(f"[Escucha] max volumen {self._d_max:.4f} (umbral {self.umbral:.4f}) | "
                  f"bloqueado por voz de ATOM: {100 * self._d_bloq / max(self._d_n, 1):.0f}% | "
                  f"desbordes: {self.desbordes} | cola: {self._cola.qsize()}")
            self._d_t, self._d_max, self._d_bloq, self._d_n = time.time(), 0.0, 0, 0

    # ── Microfono ──────────────────────────────────────────────
    def _al_recibir(self, datos, frames, tiempo, estado):
        if estado.input_overflow:
            self.desbordes += 1
        self._cola.put(datos[:, 0].copy())

    def _vaciar_cola(self):
        """Descarta audio viejo (lo grabado mientras ATOM pensaba o hablaba)."""
        try:
            while True:
                self._cola.get_nowait()
        except queue.Empty:
            pass

    def _leer_bloque(self, timeout=1.0):
        try:
            return self._cola.get(timeout=timeout)
        except queue.Empty:
            return np.zeros(BLOQUE, dtype=np.float32)

    def cerrar(self):
        self._stream.stop()
        self._stream.close()

    def calibrar_ruido(self, segundos=1.0):
        self._vaciar_cola()
        audio = np.concatenate([self._leer_bloque() for _ in range(int(segundos * SR / BLOQUE))])
        ruido = float(np.sqrt(np.mean(audio ** 2)))
        self.umbral = max(RMS_MINIMO, ruido * FACTOR_RUIDO)
        return ruido

    def _grabar_fragmento(self, max_seg, espera_max=None):
        """Espera voz y graba hasta que haya silencio. None si se agota la espera."""
        inicio = time.time()
        partes, voz_activa, silencio = [], False, 0.0
        previo = None  # bloque anterior: conserva el inicio de la palabra
        while True:
            bloque = self._leer_bloque()
            rms = float(np.sqrt(np.mean(bloque ** 2)))
            hablando = atom_esta_hablando()
            if self.debug:
                self._diag(rms, hablando)
            if hablando:
                # Es la voz de ATOM (este u otro proceso): descartar
                partes, voz_activa, silencio = [], False, 0.0
                continue
            if not voz_activa:
                if rms > self.umbral:
                    voz_activa = True
                    partes = ([previo] if previo is not None else []) + [bloque]
                elif espera_max and time.time() - inicio > espera_max:
                    return None
                previo = bloque
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
        return transcribir(modelo, audio, debug=self.debug)

    # ── API publica ────────────────────────────────────────────
    def esperar_activacion(self, timeout=None, verbose=False):
        """Bloquea hasta oir 'ATOM'. Retorna el texto oido, o None si timeout."""
        if self.umbral is None:
            self.calibrar_ruido()
        time.sleep(0.4)  # deja apagar el eco de la voz de ATOM en los altavoces
        self._vaciar_cola()
        fin = time.time() + timeout if timeout else None
        while fin is None or time.time() < fin:
            restante = (fin - time.time()) if fin else None
            audio = self._grabar_fragmento(MAX_FRAGMENTO, espera_max=restante)
            if audio is None:
                return None
            t = time.time()
            texto = self._transcribir(self._w_act, audio)
            if verbose:
                print(f"  [oído] \"{texto}\" ({len(audio) / SR:.1f}s de audio, "
                      f"transcrito en {time.time() - t:.1f}s)")
            if detectar_nombre(texto)[0]:
                # Re-transcribir con el modelo preciso: la pregunta que
                # viene tras "ATOM" importa ("como lo hay" -> "¿cómo voy?")
                if self._w_frase is not self._w_act:
                    preciso = self._transcribir(self._w_frase, audio)
                    if verbose:
                        print(f"  [preciso] \"{preciso}\"")
                    # La activacion ya la confirmo el modelo rapido; el preciso
                    # suele escribir distinto "ATOM" ("Vaya ton") pero entiende
                    # mejor la pregunta, asi que se usa si tiene texto.
                    if preciso:
                        return preciso
                return texto
        return None

    def escuchar_frase(self, max_seg=8.0, espera_max=6.0):
        """Graba la siguiente frase y la transcribe con el modelo preciso."""
        if self.umbral is None:
            self.calibrar_ruido()
        time.sleep(0.4)  # deja apagar el eco de la voz de ATOM en los altavoces
        self._vaciar_cola()
        fin = time.time() + espera_max
        while time.time() < fin:
            audio = self._grabar_fragmento(max_seg, espera_max=fin - time.time())
            if audio is None:
                return ""
            texto = self._transcribir(self._w_frase, audio)
            if texto:
                return texto
        return ""
