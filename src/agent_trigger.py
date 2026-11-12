"""
ATOM - Motor de Intervención Inteligente v2.0
==============================================
Reemplaza el sistema de disparo por umbrales fijos con un motor
híbrido que combina:
  1. Predicción ML (Random Forest) para fatiga cognitiva
  2. Heurísticas de postura cervical (pitch/yaw/roll)
  3. Fallback a umbrales fijos si no hay modelo entrenado

Mantiene compatibilidad con la API de GPT-4o-mini para generar
mensajes contextuales, y agrega alertas de ergonomía.

Changelog:
  - 2026-04-07: Integración ML + alertas de postura + cooldown inteligente
"""

import sqlite3
import pandas as pd
import time
import requests
import os
import env_loader  # carga claves desde .env
from datetime import datetime, timedelta

try:
    import pyttsx3
    def _hablar(texto):
        try:
            motor = pyttsx3.init()
            motor.setProperty('rate', 150)
            motor.setProperty('volume', 1.0)
            motor.say(texto)
            motor.runAndWait()
            motor.stop()
        except Exception as e:
            print(f"Error de audio: {e}")
except ImportError:
    def _hablar(texto):
        print(f"[TTS no disponible] {texto}")

try:
    from plyer import notification
    def _notificar(titulo, mensaje):
        try:
            notification.notify(
                title=titulo,
                message=mensaje,
                app_name="ATOM System",
                timeout=10
            )
        except Exception:
            pass
except ImportError:
    def _notificar(titulo, mensaje):
        pass

# Importar el predictor de fatiga
try:
    from fatigue_model import FatiguePredictor
except ImportError:
    FatiguePredictor = None


class SmartTriggerEngine:
    """
    Motor de intervención v2.0 con predicción ML y alertas ergonómicas.

    Tipos de intervención:
      - FATIGA COGNITIVA: Cuando el modelo ML (o heurística) indica
        alta probabilidad de fatiga
      - POSTURA CERVICAL: Cuando pitch/yaw exceden umbrales durante
        un período sostenido
      - FIJACIÓN VISUAL: Cuando la tasa de parpadeo es anormalmente
        baja (mirada fija)
    """

    def __init__(self, db_path="../data/atom_telemetry.db"):
        self.db_path = db_path

        self.check_interval = 5           # Segundos entre chequeos
        self.WINDOW_MINUTES = 1.0         # Ventana de datos a analizar
        self.FATIGUE_THRESHOLD = 0.65     # Probabilidad ML para disparar
        self.POSTURE_PITCH_LIMIT = 25.0   # Grados de inclinación frontal
        self.POSTURE_YAW_LIMIT = 30.0     # Grados de rotación lateral
        self.POSTURE_DURATION_MIN = 2.0   # Minutos sostenidos en mala postura
        self.COOLDOWN_FATIGUE = 60        # Cooldown tras alerta de fatiga (seg)
        self.COOLDOWN_POSTURE = 120       # Cooldown tras alerta de postura (seg)

        # API Key desde variable de entorno (más seguro)
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.gpt_url = "https://api.openai.com/v1/chat/completions"

        # Timestamps de última alerta (para cooldown)
        self._last_fatigue_alert = 0
        self._last_posture_alert = 0

        # Inicializar modelo ML
        self.predictor = None
        if FatiguePredictor:
            try:
                self.predictor = FatiguePredictor(db_path=db_path)
                self.predictor.compute_baseline()
            except Exception as e:
                print(f"[Trigger] ML no disponible: {e}")

    def _get_recent_data(self):
        """Lee datos recientes de ambas tablas."""
        try:
            conn = sqlite3.connect(self.db_path)
            time_window = (
                datetime.now() - timedelta(minutes=self.WINDOW_MINUTES)
            ).strftime("%Y-%m-%d %H:%M:%S")

            df_sys = pd.read_sql_query(
                f"SELECT keystrokes, mouse_clicks, active_window "
                f"FROM system_telemetry WHERE timestamp >= '{time_window}'",
                conn
            )

            # Leer métricas visuales expandidas
            try:
                df_vis = pd.read_sql_query(
                    f"SELECT avg_ear, blinks, pitch, yaw "
                    f"FROM visual_telemetry WHERE timestamp >= '{time_window}'",
                    conn
                )
            except Exception:
                # Fallback si las columnas nuevas aún no existen
                df_vis = pd.read_sql_query(
                    f"SELECT avg_ear FROM visual_telemetry "
                    f"WHERE timestamp >= '{time_window}'",
                    conn
                )

            conn.close()
            return df_sys, df_vis
        except Exception:
            return pd.DataFrame(), pd.DataFrame()

    def _check_fatigue(self, df_sys, df_vis):
        """
        Evalúa fatiga cognitiva usando ML o heurística.
        Retorna (should_alert: bool, probability: float, context: dict)
        """
        if df_sys.empty or df_vis.empty:
            return False, 0.0, {}

        metrics = {
            'avg_ear': float(df_vis['avg_ear'].mean()),
            'keystrokes': float(df_sys['keystrokes'].mean()),
            'mouse_clicks': float(df_sys['mouse_clicks'].mean()),
            'blinks': int(df_vis.get('blinks', pd.Series([0])).sum()),
            'ear_std': float(df_vis['avg_ear'].std()) if len(df_vis) > 1 else 0.02,
        }

        window_name = "Desconocida"
        if not df_sys['active_window'].empty:
            window_name = str(df_sys['active_window'].value_counts().idxmax())

        # Predecir con ML o heurística
        if self.predictor:
            probability = self.predictor.predict(metrics)
            mode = "ML" if self.predictor.is_trained else "Heurístico"
        else:
            # Fallback manual ultra-básico
            probability = 0.0
            if metrics['avg_ear'] < 0.22:
                probability += 0.5
            if metrics['keystrokes'] < 20:
                probability += 0.3
            mode = "Fallback"

        context = {
            'window_name': window_name,
            'ear': metrics['avg_ear'],
            'apm': metrics['keystrokes'],
            'probability': probability,
            'mode': mode
        }

        should_alert = (
            probability >= self.FATIGUE_THRESHOLD and
            time.time() - self._last_fatigue_alert > self.COOLDOWN_FATIGUE
        )

        return should_alert, probability, context

    def _check_posture(self, df_vis):
        """
        Evalúa si la postura cervical es mala de forma sostenida.
        Retorna (should_alert: bool, posture_info: dict)
        """
        if 'pitch' not in df_vis.columns or df_vis.empty:
            return False, {}

        avg_pitch = float(df_vis['pitch'].mean())
        avg_yaw = float(df_vis['yaw'].mean())

        bad_posture = (
            abs(avg_pitch) > self.POSTURE_PITCH_LIMIT or
            abs(avg_yaw) > self.POSTURE_YAW_LIMIT
        )

        posture_info = {
            'pitch': avg_pitch,
            'yaw': avg_yaw,
            'issue': None
        }

        if abs(avg_pitch) > self.POSTURE_PITCH_LIMIT:
            direction = "hacia la pantalla" if avg_pitch > 0 else "hacia atrás"
            posture_info['issue'] = f"Inclinación cervical {direction} ({avg_pitch:.1f}°)"
        elif abs(avg_yaw) > self.POSTURE_YAW_LIMIT:
            direction = "a la derecha" if avg_yaw > 0 else "a la izquierda"
            posture_info['issue'] = f"Rotación cervical {direction} ({avg_yaw:.1f}°)"

        should_alert = (
            bad_posture and
            time.time() - self._last_posture_alert > self.COOLDOWN_POSTURE
        )

        return should_alert, posture_info

    def _invoke_fatigue_agent(self, context):
        """Genera y emite alerta de fatiga cognitiva vía GPT."""
        print("\n" + "=" * 55)
        print(f"  🚨 ALERTA DE FATIGA COGNITIVA ({context['mode']})")
        print(f"  Probabilidad: {context['probability']:.0%}")
        print(f"  EAR: {context['ear']:.3f} | APM: {context['apm']:.1f}")
        print("=" * 55)

        self._last_fatigue_alert = time.time()

        # Generar mensaje con GPT si hay API key
        if self.api_key:
            prompt = (
                f"Eres ATOM, un asistente IA servicial de monitoreo cognitivo. "
                f"Detectaste fatiga (probabilidad: {context['probability']:.0%}) "
                f"mientras el usuario trabaja en: '{context['window_name']}'. "
                f"Su EAR es {context['ear']:.2f} (normal ~0.27). "
                f"Sugiérele un breve descanso. Tono cálido y elegante. Máximo 40 palabras."
            )
            texto = self._call_gpt(prompt)
        else:
            texto = (
                f"ATOM detecta fatiga cognitiva ({context['probability']:.0%}). "
                f"Te recomiendo una pausa de 5 minutos. "
                f"Tu concentración te lo agradecerá."
            )

        print(f"\n🤖 ATOM: \"{texto}\"")
        _notificar("🧠 ATOM - Fatiga Cognitiva", texto)
        _hablar(texto)

    def _invoke_posture_agent(self, posture_info):
        """Genera y emite alerta de postura cervical."""
        print("\n" + "=" * 55)
        print(f"  🦴 ALERTA DE POSTURA CERVICAL")
        print(f"  {posture_info['issue']}")
        print("=" * 55)

        self._last_posture_alert = time.time()

        if self.api_key:
            prompt = (
                f"Eres ATOM, un asistente IA de ergonomía. "
                f"Detectaste mala postura: {posture_info['issue']}. "
                f"Pitch: {posture_info['pitch']:.1f}°, Yaw: {posture_info['yaw']:.1f}°. "
                f"Pide al usuario que corrija su posición de forma amable. "
                f"Menciona el riesgo cervical. Máximo 35 palabras."
            )
            texto = self._call_gpt(prompt)
        else:
            texto = (
                f"ATOM detecta {posture_info['issue']}. "
                f"Por favor, ajusta tu silla y pantalla para evitar dolor cervical."
            )

        print(f"\n🤖 ATOM: \"{texto}\"")
        _notificar("🦴 ATOM - Postura Cervical", texto)
        _hablar(texto)

    def _call_gpt(self, prompt):
        """Llama a GPT-4o-mini y retorna el texto generado."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 80,
            "temperature": 0.7
        }
        try:
            response = requests.post(self.gpt_url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"[GPT] Error: {e}")
            return "ATOM recomienda una pausa breve para recuperar tu concentración."

    def run_monitor(self):
        """Bucle principal del motor de intervención."""
        print("=" * 55)
        print("  ATOM - Motor de Intervención Inteligente v2.0")
        print("=" * 55)

        if self.predictor:
            status = self.predictor.get_status_report()
            print(f"  Modo: {status['mode']}")
            print(f"  Umbral EAR: {status['threshold']:.3f}")
        else:
            print("  Modo: Fallback (umbrales fijos)")

        print(f"  Intervalo de chequeo: {self.check_interval}s")
        print(f"  API GPT: {'Configurada' if self.api_key else 'No configurada'}")
        print("=" * 55 + "\n")

        while True:
            df_sys, df_vis = self._get_recent_data()

            if not df_sys.empty and not df_vis.empty:
                # 1. Chequeo de fatiga cognitiva
                fatigue_alert, prob, context = self._check_fatigue(df_sys, df_vis)

                # Log detallado en terminal
                mode_tag = context.get('mode', '?')
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"EAR: {context.get('ear', 0):.3f} | "
                    f"APM: {context.get('apm', 0):.1f} | "
                    f"Fatiga({mode_tag}): {prob:.0%} | "
                    f"Foco: {context.get('window_name', '?')[:25]}"
                )

                if fatigue_alert:
                    self._invoke_fatigue_agent(context)
                    time.sleep(self.COOLDOWN_FATIGUE)
                    continue

                # 2. Chequeo de postura cervical
                posture_alert, posture_info = self._check_posture(df_vis)
                if posture_alert:
                    self._invoke_posture_agent(posture_info)

            time.sleep(self.check_interval)


# ─── Compatibilidad con lanzador_atom.py ───────────────────────────────
# El lanzador espera que el script se ejecute directamente
if __name__ == "__main__":
    try:
        engine = SmartTriggerEngine()
        engine.run_monitor()
    except KeyboardInterrupt:
        print("\n[Trigger] Motor detenido.")
