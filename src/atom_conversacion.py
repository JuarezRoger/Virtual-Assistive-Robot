"""
ATOM - Asistente conversacional
===============================
Di "Oye Atómica" (o "Oye ATOM") y pregunta. Claude responde en voz usando herramientas que
consultan la base de datos de telemetria:

  - estado_actual          -> fatiga, ojos, postura y actividad recientes
  - resumen_del_dia        -> tiempo trabajado, tendencia de fatiga y postura
  - historial              -> dias anteriores (resumenes + eventos importantes)
  - registrar_nasa_tlx     -> guarda una evaluacion NASA-TLX dictada por voz
  - programar_recordatorio -> "recuerdame descansar en 20 minutos"
  - recordar / olvidar     -> memoria de largo plazo (ver atom_memoria.py)

Tokens: instrucciones y herramientas cacheadas, resultados compactos, historial
de solo 3 turnos sin llamadas a herramientas, sin razonamiento extendido y
respuestas cortas por instruccion.

Modelo configurable con ATOM_MODELO en el .env (por defecto Claude Haiku 5.5: rapido para voz).
"""

import json
import os
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta

import anthropic

import env_loader  # noqa: F401  (carga ANTHROPIC_API_KEY desde .env)
import atom_memoria as mem
from atom_voz import hablar
from fatigue_model import KSS_FATIGA, MIN_ETIQUETAS

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "atom_telemetry.db")
MODELO = os.environ.get("ATOM_MODELO", "claude-haiku-5-5")
MAX_TURNOS_HISTORIAL = 3          # turnos (pregunta+respuesta) que se conservan
REINICIO_CONVERSACION = 1800      # s sin hablar -> olvida el hilo de la conversacion
SEGUIMIENTO = 15.0                # s para seguir hablando sin decir "ATOM"

# Pregunta KSS (etiquetas para el modelo personal de fatiga)
KSS_INTERVALO_MIN = 75            # entre una respuesta y la siguiente pregunta
KSS_MAX_DIA = 6
KSS_PRIMERA_MIN = 20              # tras arrancar (deja juntar 10 min de camara)
KSS_REINTENTO_MIN = 15            # si no estaba frente a la camara o no contesto
# Sin "un": "un siete" daria 1
NUMEROS = {"uno": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
           "seis": 6, "siete": 7, "ocho": 8, "nueve": 9}
# Cierra la conversacion solo si la frase esta hecha UNICAMENTE de estas palabras
# ("gracias ATOM", "eso es todo por ahora"), no "listo, ¿cómo voy?"
PALABRAS_DESPEDIDA = {"gracias", "muchas", "eso", "es", "todo", "nada", "mas", "adios",
                      "hasta", "luego", "ya", "esta", "listo", "bye", "chao", "por",
                      "ahora", "vale", "ok", "bien", "atom", "aton", "atomo", "atomica"}
INICIO_DESPEDIDA = {"gracias", "muchas", "eso", "es", "nada", "adios", "hasta",
                    "ya", "listo", "bye", "chao"}

MAX_TOKENS = 150                  # respuesta hablada: ~2 frases. Haiku no respetaba el
                                  # limite de palabras del prompt; asi no puede alargarse
MAX_TOKENS_HERRAMIENTA = 1024     # reintento si una llamada a herramienta no cupo en 150
# Haiku 5.5 razona por defecto y ese razonamiento consumia todo max_tokens
# (respuestas vacias). Para voz no hace falta: mas rapido y barato sin el.
THINKING = {"type": "disabled"}

# Fijo y sin datos variables -> se cachea junto con las herramientas
SYSTEM_PROMPT = """Eres Atómica (del sistema ATOM), asistente de voz que monitorea fatiga y postura de quien \
trabaja en su computadora (camara: apertura ocular EAR, parpadeos, postura; sensores: \
teclado, raton, ventana activa).
Todo se lee en voz alta: espanol, maximo 2 frases (<35 palabras), sin listas, markdown \
ni emojis; numeros como se dicen y redondeados. Tono calido y sereno, sin diagnosticos medicos.
Usa herramientas para preguntas sobre su estado o sus datos; nunca inventes datos.
NASA-TLX: pide las 6 escalas 0-100 que falten antes de registrar.
La transcripcion de voz puede fallar: interpreta la intencion mas probable.
Memoria: usa `recordar` solo con informacion duradera y util para cuidarle (preferencias, \
metas, horarios, molestias recurrentes, patrones). No guardes estados pasajeros, saludos ni \
datos que ya estan en los sensores. Si pide olvidar algo, usa `olvidar`."""


# ══════════════════════════════════════════════════════════════════
#  HERRAMIENTAS (consultas a la base de datos)
# ══════════════════════════════════════════════════════════════════

@contextmanager
def _conectar(solo_lectura=True):
    """Conexion que se cierra siempre; solo las escrituras explicitas confirman."""
    if solo_lectura:
        ruta = os.path.abspath(DB_PATH).replace("\\", "/")
        conn = sqlite3.connect(f"file:{ruta}?mode=ro", uri=True, timeout=5)
    else:
        conn = sqlite3.connect(DB_PATH, timeout=5)
    try:
        yield conn
        if not solo_lectura:
            conn.commit()
    finally:
        conn.close()


def _hace(minutos):
    return (datetime.now() - timedelta(minutes=minutos)).strftime("%Y-%m-%d %H:%M:%S")


def _apps_mas_usadas(c, condicion, params, n):
    """Programas mas usados (solo el nombre: el titulo de ventana no sale del equipo)."""
    tiene_app = "app" in {r[1] for r in c.execute("PRAGMA table_info(system_telemetry)")}
    conteo = {}
    for titulo, app in c.execute(
            f"SELECT active_window, {'app' if tiene_app else 'NULL'} FROM system_telemetry "
            f"WHERE {condicion}", params):
        nombre = mem.nombre_app(titulo, app)
        if nombre:
            conteo[nombre] = conteo.get(nombre, 0) + 1
    return [a for a, _ in sorted(conteo.items(), key=lambda x: -x[1])[:n]]


def estado_actual(minutos=10):
    minutos = max(1, min(int(minutos), 240))
    desde = _hace(minutos)
    with _conectar() as c:
        vis = c.execute(
            "SELECT AVG(avg_ear), SUM(blinks), AVG(pitch), AVG(yaw), COUNT(*), "
            "SUM(CASE WHEN ABS(pitch) > 25 OR ABS(yaw) > 30 THEN 1 ELSE 0 END) "
            "FROM visual_telemetry WHERE source = 'real' AND timestamp >= ?", (desde,)).fetchone()
        sis = c.execute(
            "SELECT SUM(keystrokes), SUM(mouse_clicks), AVG(cpu_usage), COUNT(*) "
            "FROM system_telemetry WHERE timestamp >= ?", (desde,)).fetchone()
        ventana = _apps_mas_usadas(c, "timestamp >= ?", (desde,), 1)

    res = {"hora_actual": datetime.now().strftime("%H:%M"), "ventana_minutos": minutos}
    ear, blinks, pitch, yaw, n_vis, n_mala = vis
    if n_vis:
        from fatigue_model import FatiguePredictor
        pred = FatiguePredictor(db_path=DB_PATH)
        # El sensor visual guarda una fila cada 10 s: minutos reales con datos
        min_con_datos = max(n_vis * 10 / 60, 1 / 6)
        blinks_min = (blinks or 0) / min_con_datos
        prob = pred.predict({"avg_ear": ear, "blinks": blinks_min,
                             "keystrokes": (sis[0] or 0) / max(sis[3] or 1, 1)})
        if pred.is_trained:
            modo = "modelo ML"
        elif pred.personal_ear:
            modo = "heuristica relativa al EAR habitual del usuario"
        else:
            modo = "heuristica con umbrales genericos (aun no hay suficientes datos reales)"
        res["camara"] = {
            "ear_promedio": round(ear, 3) if ear else None,
            "ear_habitual_del_usuario": round(pred.personal_ear, 3) if pred.personal_ear else None,
            "minutos_con_datos_de_camara": round(min_con_datos, 1),
            "parpadeos_por_minuto": round(blinks_min, 1),
            "inclinacion_cabeza_grados": round(pitch or 0, 1),
            "giro_cabeza_grados": round(yaw or 0, 1),
            "porcentaje_tiempo_mala_postura": round(100 * (n_mala or 0) / n_vis),
            "probabilidad_fatiga_pct": round(prob * 100),
            "modo_fatiga": modo,
        }
    else:
        res["camara"] = "Sin datos reales de camara en este periodo (el sensor visual no estaba activo)."
    if sis[3]:
        res["actividad"] = {
            "teclas": int(sis[0] or 0), "clics": int(sis[1] or 0),
            "cpu_promedio": round(sis[2] or 0), "aplicacion_principal": ventana[0] if ventana else None,
        }
    else:
        res["actividad"] = "Sin datos de teclado/raton en este periodo."
    return res


def resumen_del_dia():
    hoy = datetime.now().strftime("%Y-%m-%d")
    with _conectar() as c:
        primero, ultimo, filas, activas = c.execute(
            "SELECT MIN(timestamp), MAX(timestamp), COUNT(*), "
            "SUM(CASE WHEN keystrokes + mouse_clicks > 0 THEN 1 ELSE 0 END) "
            "FROM system_telemetry WHERE timestamp LIKE ?", (hoy + "%",)).fetchone()
        horas = c.execute(
            "SELECT substr(timestamp, 12, 2) h, AVG(avg_ear), SUM(blinks), COUNT(*), "
            "SUM(CASE WHEN ABS(pitch) > 25 OR ABS(yaw) > 30 THEN 1 ELSE 0 END) "
            "FROM visual_telemetry WHERE source = 'real' AND timestamp LIKE ? "
            "GROUP BY h ORDER BY h", (hoy + "%",)).fetchall()
        ventanas = _apps_mas_usadas(c, "timestamp LIKE ?", (hoy + "%",), 3)
    kss, utiles = mem.kss_hoy()

    if not filas:
        return {"fecha": hoy, "resultado": "Hoy todavia no hay datos registrados."}
    return {
        "fecha": hoy,
        "hora_actual": datetime.now().strftime("%H:%M"),
        "primer_registro": primero[11:16], "ultimo_registro": ultimo[11:16],
        # system_sensor guarda una fila cada 10 s
        "minutos_con_actividad": round((activas or 0) * 10 / 60),
        "aplicaciones_mas_usadas": ventanas,
        "por_hora_camara": [
            {"hora": f"{h}:00", "ear": round(e, 3) if e else None,
             "parpadeos_por_minuto": round((b or 0) / max(n * 10 / 60, 1), 1),
             "porcentaje_mala_postura": round(100 * (m or 0) / n)}
            for h, e, b, n, m in horas
        ] or "Sin datos reales de camara hoy.",
        "eventos_hoy": [e for e in mem.eventos_recientes(dias=0, limite=10)],
        "somnolencia_kss_hoy": [f"{t[11:16]} {v}/9" for t, v in kss],
        "etiquetas_kss_faltan_para_modelo_personal": max(0, MIN_ETIQUETAS - utiles),
    }


def registrar_kss(valor):
    r = mem.registrar_kss(valor, origen="voluntario")
    _quizas_entrenar(r["etiquetas_utiles"])
    return r


def _quizas_entrenar(etiquetas):
    """Reentrena en segundo plano al llegar al minimo y luego cada 5 etiquetas."""
    if etiquetas >= MIN_ETIQUETAS and (etiquetas - MIN_ETIQUETAS) % 5 == 0:
        from fatigue_model import FatiguePredictor
        threading.Thread(target=lambda: FatiguePredictor(DB_PATH).train_model(),
                         daemon=True).start()


def registrar_nasa_tlx(mental, fisica, temporal, rendimiento, esfuerzo, frustracion):
    vals = [mental, fisica, temporal, rendimiento, esfuerzo, frustracion]
    vals = [max(0, min(100, int(round(float(v))))) for v in vals]
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _conectar(solo_lectura=False) as c:
        c.execute("INSERT INTO nasa_tlx VALUES (?,?,?,?,?,?,?)", (ts, *vals))
        total = c.execute("SELECT COUNT(*) FROM nasa_tlx").fetchone()[0]
    mem.registrar_evento("nasa_tlx", f"compuesto {sum(vals) / 6:.0f}/100")
    return {"guardado": True, "hora": ts[11:16], "valores": vals,
            "evaluaciones_totales": total, "faltan_para_entrenar_modelo": max(0, 10 - total)}


def programar_recordatorio(minutos, mensaje):
    minutos = float(minutos)
    if not 0 < minutos <= 480:
        return {"programado": False, "error": "Solo entre 1 minuto y 8 horas."}
    t = threading.Timer(minutos * 60, hablar, args=(f"Recordatorio: {mensaje}",))
    t.daemon = True
    t.start()
    hora = (datetime.now() + timedelta(minutes=minutos)).strftime("%H:%M")
    return {"programado": True, "sonara_a_las": hora, "mensaje": mensaje}


def _num(desc):
    return {"type": "number", "description": desc}


# Descripciones cortas: cada herramienta se envia en cada llamada
HERRAMIENTAS = [
    {"name": "estado_actual",
     "description": "Estado reciente: probabilidad de fatiga, EAR vs su habitual, parpadeos/min, "
                    "postura y actividad. Para '¿como voy?', '¿estoy cansado?', '¿mi postura?'.",
     "input_schema": {"type": "object", "properties": {"minutos": {
         "type": "integer", "description": "Minutos atras (1-240, defecto 10)"}}}},
    {"name": "resumen_del_dia",
     "description": "Jornada de hoy: inicio, minutos activos, ventanas, evolucion por hora y eventos.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "historial",
     "description": "Resumenes de dias anteriores y eventos importantes. Para comparar dias o semanas.",
     "input_schema": {"type": "object", "properties": {"dias": {
         "type": "integer", "description": "Dias atras (1-90, defecto 7)"}}}},
    {"name": "registrar_nasa_tlx",
     "description": "Guarda una autoevaluacion NASA-TLX. Solo con las 6 escalas (0-100).",
     "input_schema": {"type": "object", "properties": {
         "mental": _num("Exigencia mental"), "fisica": _num("Exigencia fisica"),
         "temporal": _num("Presion de tiempo"),
         "rendimiento": _num("0 perfecto, 100 fracaso"),
         "esfuerzo": _num("Esfuerzo"), "frustracion": _num("Frustracion")},
         "required": ["mental", "fisica", "temporal", "rendimiento", "esfuerzo", "frustracion"]}},
    {"name": "registrar_kss",
     "description": "Guarda su somnolencia (escala Karolinska 1 muy alerta - 9 muy somnoliento) "
                    "cuando la dice con un numero ('estoy en un 7').",
     "input_schema": {"type": "object", "properties": {"valor": {"type": "integer"}},
                      "required": ["valor"]}},
    {"name": "programar_recordatorio",
     "description": "ATOM dira un mensaje en voz alta tras N minutos.",
     "input_schema": {"type": "object", "properties": {
         "minutos": _num("1-480"), "mensaje": {"type": "string"}},
         "required": ["minutos", "mensaje"]}},
    {"name": "recordar",
     "description": "Guarda en memoria de largo plazo UN dato duradero y util del usuario "
                    "(un dato por llamada, para poder olvidarlos por separado).",
     "input_schema": {"type": "object", "properties": {
         "contenido": {"type": "string", "description": "Frase breve en tercera persona"},
         "categoria": {"type": "string", "enum": list(mem.CATEGORIAS)},
         "importancia": {"type": "integer", "description": "1 trivial - 5 esencial"}},
         "required": ["contenido", "categoria", "importancia"]}},
    {"name": "olvidar",
     "description": "Borra un recuerdo de la memoria de largo plazo.",
     "input_schema": {"type": "object", "properties": {"contenido": {"type": "string"}},
                      "required": ["contenido"]}},
]


def _historial_tool(dias=7):
    dias = max(1, min(int(dias), 90))
    return {"dias": mem.historial(dias), "eventos": mem.eventos_recientes(dias)}


_FUNCIONES = {"estado_actual": estado_actual, "resumen_del_dia": resumen_del_dia,
              "historial": _historial_tool, "registrar_nasa_tlx": registrar_nasa_tlx,
              "registrar_kss": registrar_kss,
              "programar_recordatorio": programar_recordatorio,
              "recordar": mem.recordar, "olvidar": mem.olvidar}


def _compactar(valor):
    """Quita campos vacios recursivamente: menos tokens en cada resultado."""
    if isinstance(valor, dict):
        return {k: _compactar(v) for k, v in valor.items() if v not in (None, "", [], {})}
    if isinstance(valor, list):
        return [_compactar(v) for v in valor]
    return valor


def ejecutar_herramienta(nombre, entrada):
    try:
        resultado = _compactar(_FUNCIONES[nombre](**entrada))
        return json.dumps(resultado, ensure_ascii=False, separators=(",", ":")), False
    except Exception as e:
        return f"Error en {nombre}: {type(e).__name__}: {e}", True


# ══════════════════════════════════════════════════════════════════
#  CONVERSACION CON CLAUDE
# ══════════════════════════════════════════════════════════════════

class Cerebro:
    def __init__(self):
        self.cliente = anthropic.Anthropic()
        self.historial = []        # solo pares (pregunta, respuesta final)
        self._ultimo = 0.0
        self.uso_total = {"entrada": 0, "cache": 0, "salida": 0, "llamadas": 0}

    def _system(self):
        bloques = [{"type": "text", "text": SYSTEM_PROMPT,
                    # Punto de cache: herramientas + instrucciones (no cambian)
                    "cache_control": {"type": "ephemeral"}}]
        memorias = mem.memorias_para_prompt()
        if memorias:
            # Despues del punto de cache: si cambia no invalida lo cacheado
            bloques.append({"type": "text", "text": "Lo que recuerdas del usuario:\n" + memorias})
        return bloques

    def preguntar(self, texto, verbose=False):
        if time.time() - self._ultimo > REINICIO_CONVERSACION:
            self.historial = []
        self._ultimo = time.time()

        # Historial compacto: los turnos anteriores sin llamadas a herramientas
        mensajes = []
        for pregunta, respuesta in self.historial:
            mensajes += [{"role": "user", "content": pregunta},
                         {"role": "assistant", "content": respuesta}]
        mensajes.append({"role": "user", "content": texto})
        system = self._system()

        for _ in range(4):  # limite de rondas de herramientas
            resp = self._llamar(system, mensajes, MAX_TOKENS, verbose)
            if resp.stop_reason == "max_tokens" and any(b.type == "tool_use" for b in resp.content):
                # Llamada a herramienta cortada (p. ej. un recuerdo largo): repetir con margen
                resp = self._llamar(system, mensajes, MAX_TOKENS_HERRAMIENTA, verbose)
            mensajes.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason != "tool_use":
                break
            resultados = []
            for b in resp.content:
                if b.type == "tool_use":
                    contenido, es_error = ejecutar_herramienta(b.name, b.input)
                    if verbose:
                        print(f"  [herramienta] {b.name}({b.input}) -> {contenido[:160]}")
                    resultados.append({"type": "tool_result", "tool_use_id": b.id,
                                       "content": contenido, "is_error": es_error})
            mensajes.append({"role": "user", "content": resultados})

        respuesta = " ".join(b.text for b in resp.content if b.type == "text").strip()
        if resp.stop_reason == "max_tokens":
            # Cortada a media frase: hablar solo hasta la ultima frase completa
            fin = max(respuesta.rfind(c) for c in ".!?")
            respuesta = respuesta[:fin + 1] if fin > 0 else respuesta
        respuesta = respuesta or "Perdón, no tengo una respuesta para eso."
        self.historial = (self.historial + [(texto, respuesta)])[-MAX_TURNOS_HISTORIAL:]
        return respuesta

    def _llamar(self, system, mensajes, max_tokens, verbose):
        resp = self.cliente.messages.create(
            model=MODELO, max_tokens=max_tokens, system=system,
            tools=HERRAMIENTAS, messages=mensajes, thinking=THINKING)
        self._contar(resp.usage, verbose)
        return resp

    def _contar(self, uso, verbose):
        cache = (uso.cache_read_input_tokens or 0) + (uso.cache_creation_input_tokens or 0)
        self.uso_total["entrada"] += uso.input_tokens
        self.uso_total["cache"] += cache
        self.uso_total["salida"] += uso.output_tokens
        self.uso_total["llamadas"] += 1
        if verbose:
            print(f"  [tokens] entrada {uso.input_tokens} | cache leida "
                  f"{uso.cache_read_input_tokens or 0} creada {uso.cache_creation_input_tokens or 0} "
                  f"| salida {uso.output_tokens}")


def _texto_tras_activacion(texto):
    """'Oye ATOM, ¿cómo voy?' -> '¿cómo voy?'"""
    from atom_escucha import detectar_nombre
    encontrado, resto = detectar_nombre(texto)
    if not encontrado:
        # "ATOM" transcrito de otra forma ("¡Vaya ton! ¡Como voy!"):
        # la pregunta es lo que sigue a la primera pausa
        partes = re.split(r"[,.;:!?]", texto, maxsplit=1)
        resto = partes[1] if len(partes) > 1 else ""
    resto = re.sub(r"^[\s,.;:!¡?¿-]+", "", resto)
    if len(resto.split()) >= 2 or (resto and resto.strip("¡!¿? ").lower() in
                                   ("como voy", "cómo voy", "hola", "ayuda")):
        return resto
    return ""


def _numero_kss(texto):
    """'Como un siete' -> 7. None si no hay un numero del 1 al 9."""
    import unicodedata
    t = "".join(c for c in unicodedata.normalize("NFD", texto.lower())
                if unicodedata.category(c) != "Mn")
    for palabra in re.findall(r"\d+|[a-z]+", t):
        if palabra.isdigit() and 1 <= int(palabra) <= 9:
            return int(palabra)
        if palabra in NUMEROS:
            return NUMEROS[palabra]
    return None


def _frente_a_camara():
    """Hay datos reales de camara en los ultimos 2 minutos."""
    with _conectar() as c:
        n = c.execute("SELECT COUNT(*) FROM visual_telemetry WHERE source = 'real' "
                      "AND timestamp >= ?", (_hace(2),)).fetchone()[0]
    return n >= 6


def preguntar_kss(oido):
    """Pregunta la somnolencia KSS si toca. Retorna minutos hasta volver a intentar."""
    hoy, _ = mem.kss_hoy()
    if len(hoy) >= KSS_MAX_DIA:
        return 60
    if hoy:
        ultima = datetime.strptime(hoy[-1][0], "%Y-%m-%d %H:%M:%S")
        faltan = KSS_INTERVALO_MIN - (datetime.now() - ultima).total_seconds() / 60
        if faltan > 0:
            return faltan
    if not _frente_a_camara():
        return KSS_REINTENTO_MIN

    print("[KSS] Preguntando somnolencia")
    if hoy:
        hablar("Pregunta rápida: del uno al nueve, ¿qué tan somnoliento te sientes?")
    else:   # primera del dia: explicar la escala
        hablar("Pregunta rápida: del uno al nueve, donde uno es muy alerta y nueve "
               "muy somnoliento, ¿cómo te sientes?")
    for intento in range(2):
        texto = oido.escuchar_frase(espera_max=8)
        print(f"Tú: {texto}" if texto else "[KSS] Sin respuesta")
        if not texto:
            return KSS_REINTENTO_MIN
        valor = _numero_kss(texto)
        if valor:
            r = mem.registrar_kss(valor, origen="pregunta")
            _quizas_entrenar(r["etiquetas_utiles"])
            print(f"[KSS] {valor}/9 guardado ({r['etiquetas_utiles']} etiquetas útiles)")
            hablar(f"Anotado, {valor}." + (" Quizá te vendría bien una pausa."
                                          if valor >= KSS_FATIGA else ""))
            return KSS_INTERVALO_MIN
        if _es_despedida(texto) or re.search(r"ahora no|despu[eé]s|luego", texto, re.I):
            hablar("Va, te pregunto más tarde.")
            return 2 * KSS_REINTENTO_MIN
        if intento == 0:
            hablar("Solo dime un número del uno al nueve.")
    return KSS_REINTENTO_MIN


def _es_despedida(texto):
    import unicodedata
    t = "".join(c for c in unicodedata.normalize("NFD", texto.lower())
                if unicodedata.category(c) != "Mn")
    palabras = re.findall(r"[a-z]+", t)
    return (0 < len(palabras) <= 6 and palabras[0] in INICIO_DESPEDIDA
            and all(p in PALABRAS_DESPEDIDA for p in palabras))


def responder(cerebro, texto, verbose=True):
    if verbose:
        print(f"Tú: {texto}")
    try:
        respuesta = cerebro.preguntar(texto, verbose=verbose)
    except anthropic.AuthenticationError:
        respuesta = "Mi clave de Anthropic no es válida. Revisa el archivo punto env."
    except anthropic.APIConnectionError:
        respuesta = "No tengo conexión a internet en este momento."
    except anthropic.APIStatusError as e:
        print(f"[Claude] Error {e.status_code}: {e.message}")
        respuesta = "Tuve un problema al pensar la respuesta. Intenta de nuevo."
    if verbose:
        print(f"Atómica: {respuesta}")
    hablar(respuesta)


def main():
    from atom_escucha import Escucha
    print("=" * 55)
    print("  ATOM - Asistente de voz")
    print(f"  Modelo: {MODELO}")
    print("  Di 'Oye Atómica' (o 'ATOM') para hablar. Ctrl+C para salir.")
    print("=" * 55)
    oido = Escucha()
    oido.debug = True
    cerebro = Cerebro()
    ruido = oido.calibrar_ruido()
    print(f"[Escucha] Ruido ambiente {ruido:.4f} -> umbral de voz {oido.umbral:.4f}")
    hablar("Atómica en línea.")
    proxima_kss = time.time() + KSS_PRIMERA_MIN * 60
    while True:
        activacion = oido.esperar_activacion(timeout=max(proxima_kss - time.time(), 1),
                                             verbose=True)
        if activacion is None:      # toca la pregunta KSS
            try:
                proxima_kss = time.time() + preguntar_kss(oido) * 60
            except Exception as e:
                print(f"[KSS] Error: {e}")
                proxima_kss = time.time() + KSS_REINTENTO_MIN * 60
            continue
        pregunta = _texto_tras_activacion(activacion)
        if not pregunta:
            hablar("Dime.")
            pregunta = oido.escuchar_frase()
        # Modo conversacion: tras cada respuesta escucha SEGUIMIENTO segundos;
        # si le hablas se renueva, si no (o te despides) vuelve a esperar el nombre
        while pregunta:
            if _es_despedida(pregunta):
                print(f"Tú: {pregunta}")
                hablar("Aquí estaré.")
                break
            responder(cerebro, pregunta)
            pregunta = oido.escuchar_frase(espera_max=SEGUIMIENTO)
        print("[ATOM] Fin de la conversación, esperando el nombre.")


if __name__ == "__main__":
    import sys
    # Corre como subproceso del lanzador: sin esto la salida queda en bufer
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")

    class _Duplicar:
        """Escribe en la consola y en atom_logs_voz_FECHA.txt (local, fuera de git)
        para poder revisar despues que oyo y que respondio."""
        def __init__(self, *destinos):
            self.destinos = destinos

        def write(self, s):
            for d in self.destinos:
                d.write(s)
                d.flush()

        def flush(self):
            for d in self.destinos:
                d.flush()

    _log = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             f"atom_logs_voz_{time.strftime('%Y%m%d')}.txt"),
                "a", encoding="utf-8")
    _log.write(f"\n--- Inicio {time.strftime('%H:%M:%S')} ---\n")
    sys.stdout = _Duplicar(sys.stdout, _log)
    sys.stderr = _Duplicar(sys.stderr, _log)
    try:
        main()
    except KeyboardInterrupt:
        print("\n[ATOM] Asistente de voz detenido.")
