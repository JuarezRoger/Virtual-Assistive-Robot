"""
ATOM - Dashboard v2.0
======================
Interfaz de monitoreo cognitivo en tiempo real.

Muestra:
  - EAR, Blinks, APM, Probabilidad de fatiga
  - Postura cervical (pitch / yaw) con indicadores visuales
  - Grafica de EAR en tiempo real (ultimos 60s)
  - Estado del sensor visual (REAL / DEMO / OFFLINE)
  - NASA-TLX integrado

Changelog:
  - 2026-04-07 v2.0: Dashboard completo con postura, blinks y ML
"""

import customtkinter as ctk
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
import os

# Importar predictor de fatiga (con fallback si no esta disponible)
try:
    sys.path.insert(0, os.path.dirname(__file__))
    from fatigue_model import FatiguePredictor
    _predictor = FatiguePredictor()
    _predictor.compute_baseline()
except Exception:
    _predictor = None

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ─── Colores del tema ──────────────────────────────────────────────────
COLOR_OK      = "#2ecc71"   # Verde
COLOR_WARN    = "#f39c12"   # Naranja
COLOR_DANGER  = "#e74c3c"   # Rojo
COLOR_BG      = "#1a1a2e"
COLOR_CARD    = "#16213e"
COLOR_ACCENT  = "#3498db"


def _ear_color(ear):
    if ear < 0.20:   return COLOR_DANGER
    if ear < 0.23:   return COLOR_WARN
    return COLOR_OK

def _apm_color(apm):
    if apm < 10:  return COLOR_DANGER
    if apm < 20:  return COLOR_WARN
    return COLOR_OK

def _fatigue_color(prob):
    if prob >= 0.65: return COLOR_DANGER
    if prob >= 0.40: return COLOR_WARN
    return COLOR_OK

def _posture_color(angle, limit):
    if abs(angle) > limit:        return COLOR_DANGER
    if abs(angle) > limit * 0.7:  return COLOR_WARN
    return COLOR_OK


class MetricCard(ctk.CTkFrame):
    """Tarjeta reutilizable para mostrar una metrica con titulo, valor y subtitulo."""

    def __init__(self, parent, title, unit="", width=180, height=110, **kwargs):
        super().__init__(parent, width=width, height=height,
                         corner_radius=15, fg_color=COLOR_CARD, **kwargs)
        self.pack_propagate(False)

        ctk.CTkLabel(self, text=title,
                     font=ctk.CTkFont(size=12), text_color="gray").pack(pady=(10, 0))
        self.lbl_value = ctk.CTkLabel(self, text="--",
                                       font=ctk.CTkFont(size=30, weight="bold"),
                                       text_color="white")
        self.lbl_value.pack()
        self.lbl_sub = ctk.CTkLabel(self, text=unit,
                                     font=ctk.CTkFont(size=10), text_color="gray")
        self.lbl_sub.pack()

    def update(self, value, color=None, sub=None):
        self.lbl_value.configure(text=str(value),
                                  text_color=color or "white")
        if sub is not None:
            self.lbl_sub.configure(text=sub)


class PostureBar(ctk.CTkFrame):
    """Indicador de angulo de postura con barra de progreso."""

    def __init__(self, parent, label, limit, **kwargs):
        super().__init__(parent, fg_color=COLOR_CARD,
                         corner_radius=12, **kwargs)
        self.limit = limit

        ctk.CTkLabel(self, text=label,
                     font=ctk.CTkFont(size=12), text_color="gray").pack(pady=(8, 2))
        self.lbl_angle = ctk.CTkLabel(self, text="0.0°",
                                       font=ctk.CTkFont(size=22, weight="bold"),
                                       text_color=COLOR_OK)
        self.lbl_angle.pack()
        self.bar = ctk.CTkProgressBar(self, width=160, height=10,
                                       corner_radius=5)
        self.bar.set(0)
        self.bar.pack(padx=15, pady=(4, 8))
        self.lbl_status = ctk.CTkLabel(self, text="OK",
                                        font=ctk.CTkFont(size=10),
                                        text_color=COLOR_OK)
        self.lbl_status.pack(pady=(0, 8))

    def update(self, angle):
        color = _posture_color(angle, self.limit)
        self.lbl_angle.configure(text=f"{angle:.1f}°", text_color=color)
        progress = min(abs(angle) / self.limit, 1.0)
        self.bar.set(progress)
        self.bar.configure(progress_color=color)
        status = "ALERTA" if abs(angle) > self.limit else (
                 "PRECAUCION" if abs(angle) > self.limit * 0.7 else "OK")
        self.lbl_status.configure(text=status, text_color=color)


class AtomDashboard(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("ATOM - Monitor Cognitivo v2.0")
        self.geometry("960x780")
        self.resizable(True, True)
        self.configure(fg_color=COLOR_BG)
        self.db_path = "../data/atom_telemetry.db"
        self.tlx_window = None
        self._sensor_mode = "OFFLINE"

        self.init_db()
        self._build_ui()
        self.update_telemetry()
        self.update_chart()

    # ─────────────────────────────────────────────────────────────
    #  CONSTRUCCION DE LA UI
    # ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Titulo ──
        title_frame = ctk.CTkFrame(self, fg_color="transparent")
        title_frame.pack(fill="x", padx=20, pady=(15, 5))

        ctk.CTkLabel(title_frame, text="ATOM",
                     font=ctk.CTkFont(size=28, weight="bold"),
                     text_color=COLOR_ACCENT).pack(side="left")
        ctk.CTkLabel(title_frame, text=" Monitor Cognitivo v2.0",
                     font=ctk.CTkFont(size=18),
                     text_color="white").pack(side="left", pady=(5, 0))

        self.lbl_sensor_badge = ctk.CTkLabel(
            title_frame, text="  OFFLINE  ",
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=COLOR_DANGER, corner_radius=8, text_color="white")
        self.lbl_sensor_badge.pack(side="right", padx=5)

        ctk.CTkLabel(title_frame, text="Sensor visual:",
                     font=ctk.CTkFont(size=11), text_color="gray").pack(side="right")

        # ── Fila 1: Metricas principales ──
        row1 = ctk.CTkFrame(self, fg_color="transparent")
        row1.pack(fill="x", padx=20, pady=(5, 0))

        self.card_ear    = MetricCard(row1, "👁️  EAR",       "apertura ocular")
        self.card_blinks = MetricCard(row1, "😑  Blinks",    "por intervalo")
        self.card_apm    = MetricCard(row1, "⌨️  APM",       "teclas/min")
        self.card_fatigue = MetricCard(row1, "🧠  Fatiga",   "probabilidad ML")

        for card in (self.card_ear, self.card_blinks,
                     self.card_apm, self.card_fatigue):
            card.pack(side="left", padx=8, expand=True, fill="x")

        # ── Fila 2: Postura ──
        row2 = ctk.CTkFrame(self, fg_color="transparent")
        row2.pack(fill="x", padx=20, pady=(10, 0))

        ctk.CTkLabel(row2, text="Postura Cervical",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color="gray").pack(side="left", padx=8)

        self.bar_pitch = PostureBar(row2, "Pitch  (inclinacion frontal)", limit=25)
        self.bar_pitch.pack(side="left", padx=8, expand=True, fill="x")

        self.bar_yaw = PostureBar(row2, "Yaw  (rotacion lateral)", limit=30)
        self.bar_yaw.pack(side="left", padx=8, expand=True, fill="x")

        self.lbl_posture_summary = ctk.CTkLabel(
            row2, text="Postura\nOK",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=COLOR_OK, width=90)
        self.lbl_posture_summary.pack(side="left", padx=8)

        # ── Ventana activa ──
        self.lbl_window = ctk.CTkLabel(
            self, text="Buscando ventana activa...",
            font=ctk.CTkFont(size=13), text_color="#f1c40f", wraplength=900)
        self.lbl_window.pack(pady=(8, 0))

        # ── Grafica EAR en tiempo real ──
        frame_chart = ctk.CTkFrame(self, corner_radius=15, fg_color=COLOR_CARD)
        frame_chart.pack(fill="both", expand=True, padx=20, pady=(8, 5))

        ctk.CTkLabel(frame_chart,
                     text="Historial EAR en tiempo real  (ultimos 2 minutos)",
                     font=ctk.CTkFont(size=13, weight="bold")).pack(pady=(8, 0))

        self.fig, self.ax = plt.subplots(figsize=(9, 2.5), facecolor="#16213e")
        self.ax.set_facecolor("#0f0f1e")
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame_chart)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=5)

        # ── Fila inferior: acciones + status ──
        bottom = ctk.CTkFrame(self, fg_color="transparent")
        bottom.pack(fill="x", padx=20, pady=(0, 12))

        self.btn_tlx = ctk.CTkButton(
            bottom, text="📝  Registrar NASA-TLX",
            font=ctk.CTkFont(weight="bold"),
            fg_color="#8e44ad", hover_color="#9b59b6",
            width=220, height=36,
            command=self.open_tlx_window)
        self.btn_tlx.pack(side="left")

        self.btn_train = ctk.CTkButton(
            bottom, text="🤖  Entrenar Modelo ML",
            font=ctk.CTkFont(weight="bold"),
            fg_color="#27ae60", hover_color="#2ecc71",
            width=220, height=36,
            command=self.train_model_async)
        self.btn_train.pack(side="left", padx=10)

        self.lbl_status = ctk.CTkLabel(
            bottom, text="Sincronizando...",
            font=ctk.CTkFont(size=11), text_color="gray")
        self.lbl_status.pack(side="right")

    # ─────────────────────────────────────────────────────────────
    #  ACTUALIZACION DE DATOS
    # ─────────────────────────────────────────────────────────────

    def _read_db(self, minutes=2):
        """Lee las dos tablas de telemetria para la ventana de tiempo dada."""
        try:
            conn = sqlite3.connect(self.db_path)
            t = (datetime.now() - timedelta(minutes=minutes)).strftime("%Y-%m-%d %H:%M:%S")

            df_sys = pd.read_sql_query(
                f"SELECT keystrokes, mouse_clicks, active_window "
                f"FROM system_telemetry WHERE timestamp >= '{t}'", conn)

            # Leer visual con columnas nuevas (con fallback si no existen)
            try:
                df_vis = pd.read_sql_query(
                    f"SELECT timestamp, avg_ear, blinks, pitch, yaw "
                    f"FROM visual_telemetry WHERE timestamp >= '{t}' "
                    f"ORDER BY id DESC LIMIT 120", conn)
            except Exception:
                df_vis = pd.read_sql_query(
                    f"SELECT timestamp, avg_ear FROM visual_telemetry "
                    f"WHERE timestamp >= '{t}' ORDER BY id DESC LIMIT 120", conn)

            conn.close()
            return df_sys, df_vis
        except Exception:
            return pd.DataFrame(), pd.DataFrame()

    def update_telemetry(self):
        try:
            df_sys, df_vis = self._read_db(minutes=2)

            # ── EAR ──
            if not df_vis.empty:
                ear = float(df_vis['avg_ear'].iloc[0])
                self.card_ear.update(f"{ear:.3f}", color=_ear_color(ear))

                # Blinks
                if 'blinks' in df_vis.columns:
                    blinks = int(df_vis['blinks'].iloc[0])
                    blink_color = COLOR_WARN if blinks > 25 or blinks < 3 else COLOR_OK
                    self.card_blinks.update(str(blinks), color=blink_color)
                else:
                    self.card_blinks.update("--", sub="sin datos aun")

                # Postura
                if 'pitch' in df_vis.columns:
                    pitch = float(df_vis['pitch'].mean())
                    yaw   = float(df_vis['yaw'].mean())
                    self.bar_pitch.update(pitch)
                    self.bar_yaw.update(yaw)

                    bad = abs(pitch) > 25 or abs(yaw) > 30
                    warn = abs(pitch) > 18 or abs(yaw) > 21
                    self.lbl_posture_summary.configure(
                        text="ALERTA\nPostura" if bad else ("Precaucion\nPostura" if warn else "Postura\nOK"),
                        text_color=COLOR_DANGER if bad else (COLOR_WARN if warn else COLOR_OK)
                    )

                # Estado del sensor
                last_ts = pd.to_datetime(df_vis['timestamp'].iloc[0])
                secs_ago = (datetime.now() - last_ts).total_seconds()
                if secs_ago < 20:
                    mode = "DEMO" if 'pitch' not in df_vis.columns or df_vis['pitch'].abs().sum() < 1 else "REAL"
                else:
                    mode = "OFFLINE"
            else:
                mode = "OFFLINE"

            self._sensor_mode = mode
            badge_colors = {"REAL": COLOR_OK, "DEMO": COLOR_WARN, "OFFLINE": COLOR_DANGER}
            self.lbl_sensor_badge.configure(
                text=f"  {mode}  ", fg_color=badge_colors[mode])

            # ── APM ──
            if not df_sys.empty:
                apm = float(df_sys['keystrokes'].mean())
                self.card_apm.update(f"{apm:.0f}", color=_apm_color(apm))
                ventana = str(df_sys['active_window'].value_counts().idxmax())
                self.lbl_window.configure(text=f"🖥️  {self._clean(ventana)}")

            # ── Fatiga (ML o heuristica) ──
            if not df_vis.empty and not df_sys.empty:
                metrics = {
                    'avg_ear': float(df_vis['avg_ear'].mean()),
                    'keystrokes': float(df_sys['keystrokes'].mean()),
                    'blinks': int(df_vis['blinks'].iloc[0]) if 'blinks' in df_vis.columns else 0,
                    'mouse_clicks': float(df_sys['mouse_clicks'].mean()) if 'mouse_clicks' in df_sys.columns else 0,
                }
                if _predictor:
                    prob = _predictor.predict(metrics)
                else:
                    prob = self._heuristic_fatigue(metrics)

                mode_label = "ML" if (_predictor and _predictor.is_trained) else "heur."
                self.card_fatigue.update(
                    f"{prob:.0%}",
                    color=_fatigue_color(prob),
                    sub=f"modo {mode_label}"
                )

            self.lbl_status.configure(
                text=f"Actualizado: {datetime.now().strftime('%H:%M:%S')}",
                text_color="gray")

        except Exception as e:
            self.lbl_status.configure(text=f"Error: {e}", text_color=COLOR_DANGER)

        self.after(2000, self.update_telemetry)

    def update_chart(self):
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(
                "SELECT timestamp, avg_ear FROM visual_telemetry "
                "ORDER BY id DESC LIMIT 120", conn)
            conn.close()

            if not df.empty:
                df = df.iloc[::-1].copy()
                df['timestamp'] = pd.to_datetime(df['timestamp'])

                self.ax.clear()
                self.ax.set_facecolor("#0f0f1e")

                # Zona de alerta
                self.ax.axhspan(0.0, 0.20, alpha=0.15, color=COLOR_DANGER)
                self.ax.axhspan(0.20, 0.23, alpha=0.10, color=COLOR_WARN)

                # Linea de EAR
                self.ax.plot(df['timestamp'], df['avg_ear'],
                             color=COLOR_ACCENT, linewidth=2, zorder=3)
                self.ax.fill_between(df['timestamp'], df['avg_ear'],
                                     0.15, alpha=0.15, color=COLOR_ACCENT)

                # Umbral
                ear_thresh = _predictor.get_adaptive_threshold() if _predictor else 0.22
                self.ax.axhline(y=ear_thresh, color=COLOR_DANGER,
                                linestyle='--', linewidth=1.2, alpha=0.8,
                                label=f'Umbral ({ear_thresh:.3f})')

                self.ax.set_ylim(0.12, 0.38)
                self.ax.set_ylabel("EAR", color="gray", fontsize=9)
                self.ax.tick_params(colors="gray", labelsize=8)
                self.ax.legend(loc="upper right", fontsize=8,
                               facecolor="#16213e", labelcolor="white")

                import matplotlib.dates as mdates
                self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                self.fig.autofmt_xdate(rotation=0)
                self.fig.tight_layout(pad=0.5)
                self.canvas.draw()

        except Exception:
            pass

        self.after(3000, self.update_chart)

    # ─────────────────────────────────────────────────────────────
    #  HELPERS
    # ─────────────────────────────────────────────────────────────

    def _clean(self, name):
        name = str(name)
        if " - " in name:
            name = name.split(" - ")[-1]
        return name[:60] + "..." if len(name) > 60 else name

    def _heuristic_fatigue(self, m):
        score = 0.0
        ear = m.get('avg_ear', 0.25)
        if ear < 0.20:   score += 0.45
        elif ear < 0.22: score += 0.25
        if m.get('keystrokes', 30) < 10: score += 0.30
        elif m.get('keystrokes', 30) < 20: score += 0.15
        if m.get('blinks', 12) > 25: score += 0.15
        return min(score, 1.0)

    def train_model_async(self):
        """Entrena el modelo ML en un hilo separado para no bloquear la UI."""
        import threading
        self.btn_train.configure(text="Entrenando...", state="disabled",
                                  fg_color="gray")
        def _train():
            try:
                from fatigue_model import FatiguePredictor
                p = FatiguePredictor(db_path=self.db_path)
                result = p.train_model()
                global _predictor
                _predictor = p
                msg = f"Modelo entrenado (F1={result:.2f})" if result else "Faltan datos TLX"
            except Exception as e:
                msg = f"Error: {e}"
            self.after(0, lambda: self.btn_train.configure(
                text="Entrenar Modelo ML", state="normal",
                fg_color="#27ae60"))
            self.after(0, lambda: self.lbl_status.configure(
                text=msg, text_color=COLOR_OK))
        threading.Thread(target=_train, daemon=True).start()

    # ─────────────────────────────────────────────────────────────
    #  BASE DE DATOS
    # ─────────────────────────────────────────────────────────────

    def init_db(self):
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute('''CREATE TABLE IF NOT EXISTS nasa_tlx (
                timestamp TEXT, mental INTEGER, fisica INTEGER,
                temporal INTEGER, rendimiento INTEGER,
                esfuerzo INTEGER, frustracion INTEGER)''')
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error BD: {e}")

    # ─────────────────────────────────────────────────────────────
    #  VENTANA NASA-TLX
    # ─────────────────────────────────────────────────────────────

    def open_tlx_window(self):
        if self.tlx_window and self.tlx_window.winfo_exists():
            return
        self.tlx_window = ctk.CTkToplevel(self)
        self.tlx_window.title("Evaluacion NASA-TLX")
        self.tlx_window.geometry("460x660")
        self.tlx_window.attributes("-topmost", True)
        self.tlx_window.resizable(False, False)

        ctk.CTkLabel(self.tlx_window, text="Evaluacion NASA-TLX",
                     font=ctk.CTkFont(size=20, weight="bold")).pack(pady=15)
        ctk.CTkLabel(self.tlx_window,
                     text="Ajusta tu percepcion actual (0 = Bajo, 100 = Alto)",
                     text_color="gray").pack(pady=(0, 10))

        self.sliders = {}
        dims = [
            ("Mental",       "Exigencia Mental"),
            ("Fisica",       "Exigencia Fisica"),
            ("Temporal",     "Exigencia Temporal"),
            ("Rendimiento",  "Rendimiento (invertido)"),
            ("Esfuerzo",     "Esfuerzo"),
            ("Frustracion",  "Nivel de Frustracion"),
        ]
        for key, label in dims:
            fr = ctk.CTkFrame(self.tlx_window, fg_color="transparent")
            fr.pack(pady=8, padx=20, fill="x")
            lbl_val = ctk.CTkLabel(fr, text="50",
                                    font=ctk.CTkFont(weight="bold", size=14))
            lbl_val.pack(side="right")
            ctk.CTkLabel(fr, text=label,
                         font=ctk.CTkFont(size=13)).pack(side="left")
            sl = ctk.CTkSlider(self.tlx_window, from_=0, to=100, number_of_steps=100)
            sl.set(50)
            sl.pack(padx=20, fill="x")
            sl.configure(command=lambda v, l=lbl_val: l.configure(text=f"{int(v)}"))
            self.sliders[key] = sl

        ctk.CTkButton(self.tlx_window, text="Guardar en BD",
                      command=self.save_tlx,
                      fg_color="#2ecc71", hover_color="#27ae60",
                      height=40).pack(pady=20)

    def save_tlx(self):
        try:
            vals = [int(self.sliders[k].get())
                    for k in ["Mental","Fisica","Temporal",
                               "Rendimiento","Esfuerzo","Frustracion"]]
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                "INSERT INTO nasa_tlx VALUES (?,?,?,?,?,?,?)", (ts, *vals))
            conn.commit()
            conn.close()
            self.tlx_window.destroy()
            self.lbl_status.configure(
                text=f"NASA-TLX guardado a las {datetime.now().strftime('%H:%M:%S')}",
                text_color=COLOR_OK)
        except Exception as e:
            print(f"Error TLX: {e}")


if __name__ == "__main__":
    app = AtomDashboard()
    app.mainloop()
