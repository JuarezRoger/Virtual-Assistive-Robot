"""
ATOM - Sensor Visual Avanzado v2.1
===================================
Captura en segundo plano:
  1. Eye Aspect Ratio (EAR) - fatiga ocular
  2. Conteo de parpadeos (blinks) por intervalo
  3. Head Pose Estimation (HPE) - pitch, yaw, roll para ergonomia

Usa MediaPipe FaceMesh (468 landmarks) si esta instalado.
Si MediaPipe o la camara no estan disponibles, entra en MODO DEMO
y genera datos simulados realistas para demostraciones.

Changelog:
  - 2026-04-07: Migracion a MediaPipe, HPE, blinks, alertas de postura
  - 2026-04-07: v2.1 - Modo demo automatico si MediaPipe no esta instalado
"""

import numpy as np
import sqlite3
import time
import math
import random
from datetime import datetime

# ── Importaciones opcionales ───────────────────────────────────────────
try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False

try:
    import mediapipe as mp
    # Verificar que la API legacy de solutions este disponible
    _solutions = mp.solutions
    _face_mesh_mod = _solutions.face_mesh
    mp_face_mesh = _face_mesh_mod
    MEDIAPIPE_AVAILABLE = True
except Exception as _mp_err:
    MEDIAPIPE_AVAILABLE = False
    mp_face_mesh = None
    print(f"[ATOM Visual] MediaPipe no disponible ({type(_mp_err).__name__}: {_mp_err}). Modo demo activado.")

# ─── Indices de landmarks MediaPipe FaceMesh ──────────────────────────
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
HPE_LANDMARKS = [1, 152, 33, 263, 61, 291]

MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0),
], dtype=np.float64)

# ─── Parametros de deteccion ───────────────────────────────────────────
EAR_BLINK_THRESHOLD   = 0.21
EAR_CONSEC_FRAMES     = 2
SAVE_INTERVAL_SECONDS = 10
POSTURE_ALERT_PITCH   = 25.0
POSTURE_ALERT_YAW     = 30.0


def _compute_ear(eye_landmarks):
    p = eye_landmarks
    v1 = np.linalg.norm(p[1] - p[5])
    v2 = np.linalg.norm(p[2] - p[4])
    h  = np.linalg.norm(p[0] - p[3])
    if h == 0:
        return 0.0
    return (v1 + v2) / (2.0 * h)


def _estimate_head_pose(landmarks_2d, frame_shape):
    if not CV2_AVAILABLE:
        return 0.0, 0.0, 0.0
    h, w = frame_shape[:2]
    image_points = np.array(landmarks_2d, dtype=np.float64)
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))
    success, rotation_vec, translation_vec = cv2.solvePnP(
        MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 0.0, 0.0, 0.0
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(
        cv2.hconcat([rotation_mat, translation_vec])
    )
    return float(euler_angles[0]), float(euler_angles[1]), float(euler_angles[2])


# ══════════════════════════════════════════════════════════════════
#  GENERADOR DE DATOS DEMO
#  Simula una sesion de trabajo realista con fatiga progresiva
# ══════════════════════════════════════════════════════════════════

class DemoDataGenerator:
    """
    Genera datos sensoriales simulados que imitan una sesion de
    trabajo real con fatiga cognitiva progresiva.

    El EAR comienza alto (~0.28) y decae gradualmente.
    La postura se degrada levemente con el tiempo.
    Los blinks varian de forma natural.
    """

    def __init__(self):
        self.elapsed = 0
        self.base_ear = 0.275
        self.fatigue_rate = 0.0003  # cuanto baja el EAR por segundo
        self._blink_timer = 0
        self._in_blink = False
        self._blink_duration = 0

    def next_frame(self, dt=0.033):
        """Genera metricas para el siguiente frame (~30fps)."""
        self.elapsed += dt

        # EAR decae con el tiempo (fatiga) + ruido natural
        ear_base = self.base_ear - (self.fatigue_rate * self.elapsed)
        ear_base = max(0.18, ear_base)  # minimo fisiologico
        noise = random.gauss(0, 0.008)

        # Simular parpadeos periodicos (aprox 15-20 por minuto)
        self._blink_timer += dt
        blink_interval = random.uniform(3.0, 5.0)
        if self._blink_timer > blink_interval and not self._in_blink:
            self._in_blink = True
            self._blink_duration = random.uniform(0.10, 0.20)
            self._blink_timer = 0

        if self._in_blink:
            self._blink_duration -= dt
            ear = random.uniform(0.06, 0.12)
            if self._blink_duration <= 0:
                self._in_blink = False
        else:
            ear = max(0.15, ear_base + noise)

        # Postura: ligera deriva con el tiempo
        pitch = 3.0 + math.sin(self.elapsed / 120) * 8 + random.gauss(0, 1.5)
        yaw   = -2.0 + math.sin(self.elapsed / 90) * 5 + random.gauss(0, 1.0)
        roll  = 1.0 + random.gauss(0, 0.5)

        return ear, pitch, yaw, roll


class VisualSensorPro:
    """
    Sensor visual con soporte para modo real (MediaPipe + camara)
    y modo demo (datos simulados cuando MediaPipe no esta disponible).
    """

    def __init__(self, camera_index=1, db_path="../data/atom_telemetry.db"):
        self.db_path = db_path
        self.camera_index = camera_index

        self._ear_values   = []
        self._pitch_values = []
        self._yaw_values   = []
        self._roll_values  = []
        self._blink_count   = 0
        self._blink_counter = 0

        self.last_avg_ear = 0.0
        self.last_blinks  = 0
        self.last_pitch   = 0.0
        self.last_yaw     = 0.0
        self.last_roll    = 0.0

        self.on_posture_alert = None
        self._ensure_schema()

    def _ensure_schema(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS visual_telemetry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                avg_ear REAL
            )
        ''')
        cursor.execute("PRAGMA table_info(visual_telemetry)")
        existing_cols = {row[1] for row in cursor.fetchall()}
        new_columns = {
            'blinks': 'INTEGER DEFAULT 0',
            'pitch':  'REAL DEFAULT 0.0',
            'yaw':    'REAL DEFAULT 0.0',
            'roll':   'REAL DEFAULT 0.0'
        }
        for col_name, col_type in new_columns.items():
            if col_name not in existing_cols:
                cursor.execute(
                    f"ALTER TABLE visual_telemetry ADD COLUMN {col_name} {col_type}"
                )
                print(f"  [DB] Columna '{col_name}' agregada a visual_telemetry")
        conn.commit()
        conn.close()

    def _save_interval_to_db(self):
        if not self._ear_values:
            return
        avg_ear   = float(np.mean(self._ear_values))
        avg_pitch = float(np.mean(self._pitch_values)) if self._pitch_values else 0.0
        avg_yaw   = float(np.mean(self._yaw_values))   if self._yaw_values   else 0.0
        avg_roll  = float(np.mean(self._roll_values))   if self._roll_values  else 0.0
        blinks    = self._blink_count

        self.last_avg_ear = avg_ear
        self.last_blinks  = blinks
        self.last_pitch   = avg_pitch
        self.last_yaw     = avg_yaw
        self.last_roll    = avg_roll

        try:
            conn = sqlite3.connect(self.db_path)
            local_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conn.execute(
                "INSERT INTO visual_telemetry (timestamp, avg_ear, blinks, pitch, yaw, roll)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (local_time, avg_ear, blinks, avg_pitch, avg_yaw, avg_roll)
            )
            conn.commit()
            conn.close()
            print(f"[Visual Log] {local_time} | EAR: {avg_ear:.3f} | "
                  f"Blinks: {blinks} | Pitch: {avg_pitch:.1f} | "
                  f"Yaw: {avg_yaw:.1f} | Guardado en DB")
        except Exception as e:
            print(f"  [ERROR DB] {e}")

        if abs(avg_pitch) > POSTURE_ALERT_PITCH and self.on_posture_alert:
            self.on_posture_alert("pitch", avg_pitch)
        if abs(avg_yaw) > POSTURE_ALERT_YAW and self.on_posture_alert:
            self.on_posture_alert("yaw", avg_yaw)

        self._ear_values.clear()
        self._pitch_values.clear()
        self._yaw_values.clear()
        self._roll_values.clear()
        self._blink_count = 0

    # ─────────────────────────────────────────────────────────────
    #  MODO DEMO
    # ─────────────────────────────────────────────────────────────

    def _run_demo(self):
        """
        Modo demo: genera datos simulados sin camara ni MediaPipe.
        Util para demostraciones y pruebas de integracion.
        """
        print("=" * 55)
        print("  ATOM - Sensor Visual [MODO DEMO]")
        print("  MediaPipe no instalado. Generando datos simulados.")
        print(f"  Guardando en BD cada {SAVE_INTERVAL_SECONDS}s")
        print("  Presiona Ctrl+C para detener")
        print("=" * 55 + "\n")

        gen = DemoDataGenerator()
        last_save = time.time()

        while True:
            ear, pitch, yaw, roll = gen.next_frame()

            self._ear_values.append(ear)
            self._pitch_values.append(pitch)
            self._yaw_values.append(yaw)
            self._roll_values.append(roll)

            # Logica de blink
            if ear < EAR_BLINK_THRESHOLD:
                self._blink_counter += 1
            else:
                if self._blink_counter >= EAR_CONSEC_FRAMES:
                    self._blink_count += 1
                self._blink_counter = 0

            if time.time() - last_save >= SAVE_INTERVAL_SECONDS:
                self._save_interval_to_db()
                last_save = time.time()

            time.sleep(0.033)  # ~30 fps simulados

    # ─────────────────────────────────────────────────────────────
    #  MODO REAL (MediaPipe + camara)
    # ─────────────────────────────────────────────────────────────

    def _draw_overlay(self, frame, ear, pitch, yaw, roll):
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (330, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        ear_color   = (0, 255, 0) if ear > EAR_BLINK_THRESHOLD else (0, 0, 255)
        pitch_color = (0, 0, 255) if abs(pitch) > POSTURE_ALERT_PITCH else (0, 255, 0)
        yaw_color   = (0, 0, 255) if abs(yaw) > POSTURE_ALERT_YAW else (0, 255, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"EAR: {ear:.3f}",             (20, 35),  font, 0.6, ear_color, 2)
        cv2.putText(frame, f"Blinks: {self._blink_count}", (20, 60),  font, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"Pitch: {pitch:.1f} deg",     (20, 85),  font, 0.6, pitch_color, 2)
        cv2.putText(frame, f"Yaw: {yaw:.1f} deg",         (20, 110), font, 0.6, yaw_color, 2)
        cv2.putText(frame, f"Roll: {roll:.1f} deg",        (20, 135), font, 0.6, (255,255,255), 2)
        cv2.putText(frame, "ATOM HPE v2.1",                (20, 155), font, 0.4, (128,128,128), 1)

    def _run_real(self):
        """Modo real con camara y MediaPipe."""
        cap = cv2.VideoCapture(self.camera_index)
        print("=" * 55)
        print("  ATOM - Sensor Visual v2.1 (MediaPipe + HPE + Blinks)")
        print("=" * 55)

        if not cap.isOpened():
            print(f"[ATOM Visual] ERROR: No se pudo acceder a la camara {self.camera_index}.")
            print("[ATOM Visual] Cambiando a MODO DEMO automaticamente...")
            cap.release()
            self._run_demo()
            return

        print(f"[ATOM Visual] Camara {self.camera_index} conectada.")
        print(f"[ATOM Visual] Guardando cada {SAVE_INTERVAL_SECONDS}s. Presiona 'q' para salir.\n")

        last_save_time = time.time()

        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)

                pitch, yaw, roll, ear = 0.0, 0.0, 0.0, 0.0

                if results.multi_face_landmarks:
                    face = results.multi_face_landmarks[0]
                    h, w = frame.shape[:2]
                    landmarks = np.array([
                        (face.landmark[i].x * w, face.landmark[i].y * h)
                        for i in range(468)
                    ])
                    left_pts  = np.array([landmarks[i] for i in LEFT_EYE])
                    right_pts = np.array([landmarks[i] for i in RIGHT_EYE])
                    ear = (_compute_ear(left_pts) + _compute_ear(right_pts)) / 2.0
                    self._ear_values.append(ear)

                    if ear < EAR_BLINK_THRESHOLD:
                        self._blink_counter += 1
                    else:
                        if self._blink_counter >= EAR_CONSEC_FRAMES:
                            self._blink_count += 1
                        self._blink_counter = 0

                    hpe_points = [landmarks[i] for i in HPE_LANDMARKS]
                    pitch, yaw, roll = _estimate_head_pose(hpe_points, frame.shape)
                    self._pitch_values.append(pitch)
                    self._yaw_values.append(yaw)
                    self._roll_values.append(roll)
                    self._draw_overlay(frame, ear, pitch, yaw, roll)

                cv2.imshow("ATOM Visual Pro", frame)

                if time.time() - last_save_time >= SAVE_INTERVAL_SECONDS:
                    self._save_interval_to_db()
                    last_save_time = time.time()

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self._save_interval_to_db()
        cap.release()
        cv2.destroyAllWindows()
        print("[ATOM Visual] Sensor detenido.")

    # ─────────────────────────────────────────────────────────────
    #  PUNTO DE ENTRADA PRINCIPAL
    # ─────────────────────────────────────────────────────────────

    def run(self):
        """
        Arranca el sensor eligiendo automaticamente el modo correcto:
          - REAL:  MediaPipe instalado + camara disponible
          - DEMO:  MediaPipe no instalado (datos simulados realistas)
        """
        if not MEDIAPIPE_AVAILABLE:
            print("[ATOM Visual] MediaPipe no encontrado. Entrando en MODO DEMO.")
            print("[ATOM Visual] Para activar modo real: pip install mediapipe==0.10.21\n")
            self._run_demo()
        else:
            self._run_real()

    def get_current_metrics(self):
        return {
            'avg_ear': self.last_avg_ear,
            'blinks':  self.last_blinks,
            'pitch':   self.last_pitch,
            'yaw':     self.last_yaw,
            'roll':    self.last_roll
        }

    def save_to_db(self, ear):
        """Compatibilidad retroactiva."""
        local_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO visual_telemetry (timestamp, avg_ear) VALUES (?, ?)",
            (local_time, ear)
        )
        conn.commit()
        conn.close()


if __name__ == "__main__":
    sensor = VisualSensorPro(camera_index=1)
    sensor.run()
