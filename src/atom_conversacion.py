"""
ATOM - Asistente conversacional
===============================
Di "Oye ATOM" y pregunta. Claude responde en voz usando herramientas que
consultan la base de datos de telemetria:

  - estado_actual          -> fatiga, ojos, postura y actividad recientes
  - resumen_del_dia        -> tiempo trabajado, tendencia de fatiga y postura
  - registrar_nasa_tlx     -> guarda una evaluacion NASA-TLX dictada por voz
  - programar_recordatorio -> "recuerdame descansar en 20 minutos"

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
from atom_voz import hablar

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "atom_telemetry.db")
MODELO = os.environ.get("ATOM_MODELO", "claude-haiku-5-5")
MAX_TURNOS_HISTORIAL = 20         # mensajes que se conservan
REINICIO_CONVERSACION = 300       # s sin hablar -> nueva conversacion
SEGUIMIENTO = 6.0                 # s para seguir hablando sin decir "ATOM"

SYSTEM_PROMPT = """Eres ATOM, un asistente de voz de monitoreo cognitivo y ergonomico que \
acompana a una persona mientras trabaja en su computadora. Una camara mide la apertura \
de sus ojos (EAR), parpadeos y postura de cabeza; otro sensor mide teclado, raton y \
ventana activa. Tienes herramientas para consultar esos datos.

Reglas de respuesta (todo lo que escribes se lee en voz alta):
- Responde en espanol, en maximo 2 frases cortas (menos de 35 palabras): se escucha, no se lee. \
Si hay matices, da lo esencial y ofrece ampliar. Nada de listas, markdown, emojis ni simbolos.
- Escribe los numeros como se dicen ("veinte minutos", "unas dos horas"); redondea.
- Usa las herramientas cuando la pregunta sea sobre su estado, su jornada o sus datos. \
No inventes datos: si una herramienta indica que no hay datos reales, dilo con sencillez.
- Tono calido, sereno y profesional. No des diagnosticos medicos.
- Para registrar un NASA-TLX necesitas las seis escalas de 0 a 100 (mental, fisica, \
temporal, rendimiento, esfuerzo, frustracion). Si faltan, pide las que falten.
- La transcripcion de voz puede tener errores; interpreta la intencion mas probable."""


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
        ventana = c.execute(
            "SELECT active_window, COUNT(*) n FROM system_telemetry WHERE timestamp >= ? "
            "GROUP BY active_window ORDER BY n DESC LIMIT 1", (desde,)).fetchone()

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
            "probabilidad_fatiga": round(prob, 2),
            "modo_fatiga": modo,
        }
    else:
        res["camara"] = "Sin datos reales de camara en este periodo (el sensor visual no estaba activo)."
    if sis[3]:
        res["actividad"] = {
            "teclas": int(sis[0] or 0), "clics": int(sis[1] or 0),
            "cpu_promedio": round(sis[2] or 0), "ventana_principal": ventana[0] if ventana else None,
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
        ventanas = c.execute(
            "SELECT active_window, COUNT(*) n FROM system_telemetry WHERE timestamp LIKE ? "
            "GROUP BY active_window ORDER BY n DESC LIMIT 3", (hoy + "%",)).fetchall()
        tlx = c.execute("SELECT COUNT(*) FROM nasa_tlx").fetchone()[0]

    if not filas:
        return {"fecha": hoy, "resultado": "Hoy todavia no hay datos registrados."}
    return {
        "fecha": hoy,
        "hora_actual": datetime.now().strftime("%H:%M"),
        "primer_registro": primero[11:16], "ultimo_registro": ultimo[11:16],
        # system_sensor guarda una fila cada 10 s
        "minutos_con_actividad": round((activas or 0) * 10 / 60),
        "ventanas_mas_usadas": [v[0] for v in ventanas],
        "por_hora_camara": [
            {"hora": f"{h}:00", "ear": round(e, 3) if e else None,
             "parpadeos_por_minuto": round((b or 0) / max(n * 10 / 60, 1), 1),
             "porcentaje_mala_postura": round(100 * (m or 0) / n)}
            for h, e, b, n, m in horas
        ] or "Sin datos reales de camara hoy.",
        "evaluaciones_nasa_tlx_totales": tlx,
        "nota": "Se necesitan al menos 10 evaluaciones NASA-TLX para entrenar el modelo ML.",
    }


def registrar_nasa_tlx(mental, fisica, temporal, rendimiento, esfuerzo, frustracion):
    vals = [mental, fisica, temporal, rendimiento, esfuerzo, frustracion]
    vals = [max(0, min(100, int(round(float(v))))) for v in vals]
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _conectar(solo_lectura=False) as c:
        c.execute("INSERT INTO nasa_tlx VALUES (?,?,?,?,?,?,?)", (ts, *vals))
        total = c.execute("SELECT COUNT(*) FROM nasa_tlx").fetchone()[0]
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


HERRAMIENTAS = [
    {
        "name": "estado_actual",
        "description": "Estado reciente del usuario: probabilidad de fatiga, apertura de ojos "
                       "(EAR) comparada con su valor habitual, parpadeos por minuto, postura de "
                       "cabeza y actividad de teclado/raton. Usala para '¿como voy?', '¿estoy "
                       "cansado?', '¿como esta mi postura?'.",
        "input_schema": {"type": "object", "properties": {
            "minutos": {"type": "integer", "description": "Minutos hacia atras a analizar (1-240). Por defecto 10."}},
            "required": []},
    },
    {
        "name": "resumen_del_dia",
        "description": "Resumen de la jornada de hoy: hora de inicio, minutos con actividad, "
                       "ventanas mas usadas y evolucion por hora de ojos, parpadeo y postura.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "registrar_nasa_tlx",
        "description": "Guarda una autoevaluacion NASA-TLX de carga de trabajo. Las seis escalas "
                       "van de 0 a 100. Solo llamala cuando el usuario haya dado los seis valores.",
        "input_schema": {"type": "object", "properties": {
            k: {"type": "number", "description": d} for k, d in [
                ("mental", "Exigencia mental"), ("fisica", "Exigencia fisica"),
                ("temporal", "Presion de tiempo"), ("rendimiento", "Que tan mal sintio su rendimiento (0 perfecto, 100 fracaso)"),
                ("esfuerzo", "Esfuerzo necesario"), ("frustracion", "Nivel de frustracion")]},
            "required": ["mental", "fisica", "temporal", "rendimiento", "esfuerzo", "frustracion"]},
    },
    {
        "name": "programar_recordatorio",
        "description": "Programa un recordatorio que ATOM dira en voz alta despues de N minutos.",
        "input_schema": {"type": "object", "properties": {
            "minutos": {"type": "number", "description": "Minutos desde ahora (1-480)."},
            "mensaje": {"type": "string", "description": "Lo que ATOM dira, breve."}},
            "required": ["minutos", "mensaje"]},
    },
]

_FUNCIONES = {"estado_actual": estado_actual, "resumen_del_dia": resumen_del_dia,
              "registrar_nasa_tlx": registrar_nasa_tlx, "programar_recordatorio": programar_recordatorio}


def ejecutar_herramienta(nombre, entrada):
    try:
        resultado = _FUNCIONES[nombre](**entrada)
        return json.dumps(resultado, ensure_ascii=False), False
    except Exception as e:
        return f"Error ejecutando {nombre}: {type(e).__name__}: {e}", True


# ══════════════════════════════════════════════════════════════════
#  CONVERSACION CON CLAUDE
# ══════════════════════════════════════════════════════════════════

class Cerebro:
    def __init__(self):
        self.cliente = anthropic.Anthropic()
        self.mensajes = []
        self._ultimo = 0.0

    def preguntar(self, texto, verbose=False):
        if time.time() - self._ultimo > REINICIO_CONVERSACION:
            self.mensajes = []
        self._ultimo = time.time()
        self.mensajes.append({"role": "user", "content": texto})

        for _ in range(6):  # limite de rondas de herramientas
            resp = self.cliente.messages.create(
                model=MODELO,
                max_tokens=1024,          # respuestas cortas para voz
                system=SYSTEM_PROMPT,
                tools=HERRAMIENTAS,
                messages=self.mensajes,
                cache_control={"type": "ephemeral"},
            )
            self.mensajes.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason != "tool_use":
                break
            resultados = []
            for b in resp.content:
                if b.type == "tool_use":
                    contenido, es_error = ejecutar_herramienta(b.name, b.input)
                    if verbose:
                        print(f"  [herramienta] {b.name}({b.input}) -> {contenido[:200]}")
                    resultados.append({"type": "tool_result", "tool_use_id": b.id,
                                       "content": contenido, "is_error": es_error})
            self.mensajes.append({"role": "user", "content": resultados})

        self._recortar_historial()
        respuesta = " ".join(b.text for b in resp.content if b.type == "text").strip()
        return respuesta or "Perdón, no tengo una respuesta para eso."

    def _recortar_historial(self):
        # Cortar solo en un mensaje de usuario con texto (no tool_result)
        while len(self.mensajes) > MAX_TURNOS_HISTORIAL:
            self.mensajes.pop(0)
            while self.mensajes and not (self.mensajes[0]["role"] == "user"
                                         and isinstance(self.mensajes[0]["content"], str)):
                self.mensajes.pop(0)


def _texto_tras_activacion(texto):
    """'Oye ATOM, ¿cómo voy?' -> '¿cómo voy?'"""
    from atom_escucha import PATRON_ACTIVACION
    m = PATRON_ACTIVACION.search(texto)
    resto = texto[m.end():] if m else ""
    resto = re.sub(r"^[\s,.;:!¡?¿-]+", "", resto)
    return resto if len(resto.split()) >= 2 else ""


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
        print(f"ATOM: {respuesta}")
    hablar(respuesta)


def main():
    from atom_escucha import Escucha
    print("=" * 55)
    print("  ATOM - Asistente de voz")
    print(f"  Modelo: {MODELO}")
    print("  Di 'Oye ATOM' para hablar. Ctrl+C para salir.")
    print("=" * 55)
    oido = Escucha()
    cerebro = Cerebro()
    oido.calibrar_ruido()
    hablar("ATOM en línea.")
    while True:
        activacion = oido.esperar_activacion()
        pregunta = _texto_tras_activacion(activacion)
        if not pregunta:
            hablar("Dime.")
            pregunta = oido.escuchar_frase()
        # Modo conversacion: seguir sin repetir "ATOM" mientras haya respuesta
        while pregunta:
            responder(cerebro, pregunta)
            pregunta = oido.escuchar_frase(espera_max=SEGUIMIENTO)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[ATOM] Asistente de voz detenido.")
