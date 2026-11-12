"""
ATOM - Modelo Predictivo de Fatiga v1.0
========================================
Reemplaza el sistema de umbrales fijos (EAR < 0.22) con un modelo
de Random Forest personalizado que aprende TUS patrones de fatiga.

El modelo se entrena con datos fusionados de:
  - Sistema: keystrokes (APM), mouse_clicks, cpu_usage
  - Visual: avg_ear, blinks, pitch, yaw
  - Subjetivo: NASA-TLX (etiquetas de fatiga real)

Flujo:
  1. train_model()  → Entrena con datos históricos + NASA-TLX
  2. predict()      → Dado un vector de métricas, retorna probabilidad de fatiga
  3. get_threshold() → Umbral EAR adaptativo basado en tu historial personal

Changelog:
  - 2026-04-07: Implementación inicial con Random Forest
"""

import sqlite3
import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Ruta del modelo entrenado
MODEL_PATH = "../data/fatigue_model.pkl"
SCALER_PATH = "../data/fatigue_scaler.pkl"

# Umbral por defecto (fallback si no hay modelo entrenado)
DEFAULT_EAR_THRESHOLD = 0.22
DEFAULT_APM_THRESHOLD = 20

# Columnas de features para el modelo
FEATURE_COLUMNS = [
    'avg_ear',        # Apertura ocular promedio
    'ear_std',        # Variabilidad del EAR (parpadeo errático)
    'blinks_rate',    # Tasa de parpadeos por minuto
    'keystrokes',     # Actividad de teclado
    'mouse_clicks',   # Actividad de ratón
    'ear_delta',      # Cambio respecto al baseline de la mañana
    'apm_delta',      # Cambio de APM respecto al baseline
]


class FatiguePredictor:
    """
    Motor de predicción de fatiga cognitiva usando Random Forest.

    Modos de operación:
      - Con modelo entrenado: usa predict_proba() para probabilidad continua
      - Sin modelo (fallback): usa umbrales fijos como antes
      - Modo híbrido: combina predicción ML con heurísticas de postura
    """

    def __init__(self, db_path="../data/atom_telemetry.db"):
        self.db_path = db_path
        self.model = None
        self.scaler = None
        self.is_trained = False

        # Baselines personales (se calculan con datos de la mañana)
        self.baseline_ear = None
        self.baseline_apm = None

        # Intentar cargar modelo existente
        self._load_model()

    def _load_model(self):
        """Carga un modelo previamente entrenado, si existe."""
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            try:
                with open(MODEL_PATH, 'rb') as f:
                    self.model = pickle.load(f)
                with open(SCALER_PATH, 'rb') as f:
                    self.scaler = pickle.load(f)
                self.is_trained = True
                print("[ML] Modelo de fatiga cargado exitosamente.")
            except Exception as e:
                print(f"[ML] Error cargando modelo: {e}. Usando fallback.")
                self.is_trained = False
        else:
            print("[ML] No hay modelo entrenado. Usando umbrales fijos como fallback.")

    def compute_baseline(self):
        """
        Calcula el baseline personal del usuario usando datos de las
        primeras 2 horas de actividad del día actual.
        Esto permite detectar fatiga relativa al rendimiento matutino.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            today = datetime.now().strftime("%Y-%m-%d")
            morning_end = f"{today} 10:00:00"

            df_vis = pd.read_sql_query(
                f"SELECT avg_ear FROM visual_telemetry "
                f"WHERE timestamp LIKE '{today}%' AND timestamp <= '{morning_end}'",
                conn
            )
            df_sys = pd.read_sql_query(
                f"SELECT keystrokes FROM system_telemetry "
                f"WHERE timestamp LIKE '{today}%' AND timestamp <= '{morning_end}'",
                conn
            )
            conn.close()

            if not df_vis.empty:
                self.baseline_ear = float(df_vis['avg_ear'].mean())
                print(f"[ML] Baseline EAR matutino: {self.baseline_ear:.3f}")

            if not df_sys.empty:
                self.baseline_apm = float(df_sys['keystrokes'].mean())
                print(f"[ML] Baseline APM matutino: {self.baseline_apm:.1f}")

        except Exception as e:
            print(f"[ML] Error calculando baseline: {e}")

    def _prepare_training_data(self):
        """
        Construye el dataset de entrenamiento fusionando las tres fuentes
        de datos y generando etiquetas de fatiga a partir del NASA-TLX.

        Etiqueta de fatiga: 1 si el score TLX ponderado > 60, 0 si no.
        """
        conn = sqlite3.connect(self.db_path)

        try:
            df_sys = pd.read_sql_query("SELECT * FROM system_telemetry", conn)
            df_vis = pd.read_sql_query("SELECT * FROM visual_telemetry", conn)
            df_tlx = pd.read_sql_query("SELECT * FROM nasa_tlx", conn)
        except Exception as e:
            print(f"[ML] Error leyendo datos: {e}")
            conn.close()
            return None, None
        conn.close()

        if df_sys.empty or df_vis.empty or df_tlx.empty:
            print("[ML] Datos insuficientes para entrenar.")
            print(f"     Sistema: {len(df_sys)} filas | Visual: {len(df_vis)} | TLX: {len(df_tlx)}")
            return None, None

        # Convertir timestamps
        df_sys['timestamp'] = pd.to_datetime(df_sys['timestamp'])
        df_vis['timestamp'] = pd.to_datetime(df_vis['timestamp'])
        df_tlx['timestamp'] = pd.to_datetime(df_tlx['timestamp'])

        # Remuestrear a bloques de 1 minuto
        df_sys.set_index('timestamp', inplace=True)
        df_vis.set_index('timestamp', inplace=True)

        sys_1m = df_sys.resample('1min').agg({
            'keystrokes': 'sum',
            'mouse_clicks': 'sum',
            'cpu_usage': 'mean'
        }).dropna()

        # Agregar columnas faltantes con valores por defecto si no existen
        for col in ['blinks', 'pitch', 'yaw']:
            if col not in df_vis.columns:
                df_vis[col] = 0.0

        vis_1m = df_vis.resample('1min').agg({
            'avg_ear': 'mean',
            'blinks': 'sum',
        }).dropna()

        # Calcular features derivadas
        vis_1m['ear_std'] = df_vis['avg_ear'].resample('1min').std().fillna(0)

        # Fusionar sistema + visual
        df_fused = pd.merge(sys_1m, vis_1m, left_index=True, right_index=True)

        if df_fused.empty:
            print("[ML] No hay datos superpuestos entre sistema y visual.")
            return None, None

        # Calcular deltas respecto al baseline (o al promedio si no hay baseline)
        global_ear_mean = df_fused['avg_ear'].mean()
        global_apm_mean = df_fused['keystrokes'].mean()

        df_fused['ear_delta'] = df_fused['avg_ear'] - (self.baseline_ear or global_ear_mean)
        df_fused['apm_delta'] = df_fused['keystrokes'] - (self.baseline_apm or global_apm_mean)
        df_fused['blinks_rate'] = df_fused['blinks']  # ya es por minuto

        # ── Generar etiquetas de fatiga desde NASA-TLX ──
        # Cada registro TLX marca el estado de fatiga para la ventana de ±15 min
        df_fused['fatigue_label'] = 0  # Default: no fatigado

        for _, tlx_row in df_tlx.iterrows():
            tlx_time = tlx_row['timestamp']
            # Score compuesto TLX (ponderado: mental y esfuerzo pesan más)
            tlx_score = (
                tlx_row['mental'] * 0.25 +
                tlx_row['esfuerzo'] * 0.20 +
                tlx_row['temporal'] * 0.15 +
                tlx_row['frustracion'] * 0.15 +
                tlx_row['fisica'] * 0.10 +
                (100 - tlx_row['rendimiento']) * 0.15  # Invertido: alto rendimiento = baja fatiga
            )

            window_start = tlx_time - timedelta(minutes=15)
            window_end = tlx_time + timedelta(minutes=15)
            mask = (df_fused.index >= window_start) & (df_fused.index <= window_end)

            if tlx_score > 60:
                df_fused.loc[mask, 'fatigue_label'] = 1
            else:
                df_fused.loc[mask, 'fatigue_label'] = 0

        # Seleccionar features
        feature_cols = [c for c in FEATURE_COLUMNS if c in df_fused.columns]
        X = df_fused[feature_cols].fillna(0)
        y = df_fused['fatigue_label']

        print(f"[ML] Dataset preparado: {len(X)} muestras, "
              f"{y.sum()} fatigadas, {(~y.astype(bool)).sum()} normales")
        return X, y

    def train_model(self):
        """
        Entrena el Random Forest con los datos disponibles.
        Retorna las métricas de validación cruzada.
        """
        print("\n" + "=" * 50)
        print("  ATOM ML - Entrenando modelo de fatiga...")
        print("=" * 50)

        self.compute_baseline()
        X, y = self._prepare_training_data()

        if X is None or len(X) < 10:
            print("[ML] Se necesitan al menos 10 muestras para entrenar.")
            print("[ML] Tip: Registra más evaluaciones NASA-TLX durante tus sesiones.")
            return None

        # Verificar que hay ambas clases
        if y.nunique() < 2:
            print("[ML] Se necesitan datos de ambos estados (fatigado y normal).")
            print("[ML] Tip: Registra NASA-TLX cuando estés descansado Y cuando estés cansado.")
            return None

        # Escalar features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Random Forest con hiperparámetros conservadores para pocos datos
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,              # Evitar overfitting con pocos datos
            min_samples_split=5,
            min_samples_leaf=3,
            class_weight='balanced',  # Compensar desbalance de clases
            random_state=42
        )

        # Validación cruzada (k=3 por tamaño pequeño de datos)
        k_folds = min(3, len(X))
        if k_folds >= 2:
            scores = cross_val_score(self.model, X_scaled, y, cv=k_folds, scoring='f1')
            print(f"[ML] Validación cruzada (k={k_folds}): F1 = {scores.mean():.3f} ± {scores.std():.3f}")

        # Entrenar con todos los datos
        self.model.fit(X_scaled, y)
        self.is_trained = True

        # Importancia de features
        feature_cols = [c for c in FEATURE_COLUMNS if c in X.columns]
        importances = sorted(
            zip(feature_cols, self.model.feature_importances_),
            key=lambda x: x[1], reverse=True
        )
        print("\n[ML] Importancia de features:")
        for feat, imp in importances:
            bar = "█" * int(imp * 40)
            print(f"  {feat:15s} {imp:.3f} {bar}")

        # Guardar modelo
        self._save_model()

        print(f"\n[ML] Modelo entrenado y guardado exitosamente.")
        return scores.mean() if k_folds >= 2 else None

    def _save_model(self):
        """Persiste el modelo y scaler a disco."""
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(self.model, f)
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(self.scaler, f)

    def predict(self, metrics_dict):
        """
        Predice la probabilidad de fatiga dado un diccionario de métricas.

        Args:
            metrics_dict: dict con claves avg_ear, blinks, keystrokes, etc.

        Returns:
            float: probabilidad de fatiga (0.0 a 1.0)
                   Si no hay modelo, usa heurística basada en umbrales.
        """
        if not self.is_trained:
            return self._heuristic_prediction(metrics_dict)

        # Construir vector de features
        features = {}
        for col in FEATURE_COLUMNS:
            if col == 'ear_delta':
                baseline = self.baseline_ear or DEFAULT_EAR_THRESHOLD + 0.05
                features[col] = metrics_dict.get('avg_ear', 0.25) - baseline
            elif col == 'apm_delta':
                baseline = self.baseline_apm or DEFAULT_APM_THRESHOLD
                features[col] = metrics_dict.get('keystrokes', 0) - baseline
            elif col == 'blinks_rate':
                features[col] = metrics_dict.get('blinks', 0)
            elif col == 'ear_std':
                features[col] = metrics_dict.get('ear_std', 0.02)
            else:
                features[col] = metrics_dict.get(col, 0)

        X = pd.DataFrame([features])
        X_scaled = self.scaler.transform(X)

        # Probabilidad de la clase "fatigado" (clase 1)
        proba = self.model.predict_proba(X_scaled)[0][1]
        return float(proba)

    def _heuristic_prediction(self, metrics):
        """
        Fallback: calcula una probabilidad pseudo-basada en umbrales fijos.
        Útil cuando aún no hay modelo entrenado.
        """
        score = 0.0

        ear = metrics.get('avg_ear', 0.25)
        if ear < 0.20:
            score += 0.4
        elif ear < 0.22:
            score += 0.25
        elif ear < 0.24:
            score += 0.1

        apm = metrics.get('keystrokes', 30)
        if apm < 10:
            score += 0.3
        elif apm < 20:
            score += 0.2
        elif apm < 30:
            score += 0.05

        blinks = metrics.get('blinks', 0)
        if blinks > 25:  # Parpadeo excesivo
            score += 0.15
        elif blinks < 5:  # Fijación prolongada
            score += 0.1

        return min(score, 1.0)

    def get_adaptive_threshold(self):
        """
        Calcula un umbral EAR personalizado basado en el historial.
        En lugar de usar 0.22 para todos, usa percentil 15 del EAR personal.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(
                "SELECT avg_ear FROM visual_telemetry ORDER BY id DESC LIMIT 1000",
                conn
            )
            conn.close()

            if len(df) > 50:
                personal_threshold = float(np.percentile(df['avg_ear'].dropna(), 15))
                print(f"[ML] Umbral EAR personalizado: {personal_threshold:.3f} "
                      f"(vs. genérico {DEFAULT_EAR_THRESHOLD})")
                return personal_threshold
        except Exception:
            pass

        return DEFAULT_EAR_THRESHOLD

    def get_status_report(self):
        """Retorna un resumen del estado del modelo para el dashboard."""
        if self.is_trained:
            return {
                'mode': 'ML (Random Forest)',
                'baseline_ear': self.baseline_ear,
                'baseline_apm': self.baseline_apm,
                'threshold': self.get_adaptive_threshold()
            }
        else:
            return {
                'mode': 'Heurístico (umbrales fijos)',
                'baseline_ear': None,
                'baseline_apm': None,
                'threshold': DEFAULT_EAR_THRESHOLD
            }


# ─── Ejecución directa: entrenar modelo ───────────────────────────────
if __name__ == "__main__":
    print("🧠 ATOM - Entrenamiento del Modelo de Fatiga")
    print("=" * 50)

    predictor = FatiguePredictor()
    result = predictor.train_model()

    if result:
        print(f"\n✅ Modelo listo. F1-Score: {result:.3f}")

        # Prueba rápida con datos simulados
        test_cases = [
            {"avg_ear": 0.28, "blinks": 15, "keystrokes": 50, "mouse_clicks": 10,
             "label": "Normal (alerta)"},
            {"avg_ear": 0.19, "blinks": 30, "keystrokes": 5, "mouse_clicks": 2,
             "label": "Fatigado"},
            {"avg_ear": 0.23, "blinks": 8, "keystrokes": 25, "mouse_clicks": 5,
             "label": "Zona gris"},
        ]

        print("\n--- Pruebas de predicción ---")
        for case in test_cases:
            prob = predictor.predict(case)
            print(f"  {case['label']:20s} → Probabilidad fatiga: {prob:.1%}")
    else:
        print("\n⚠️  No se pudo entrenar. Necesitas más datos de NASA-TLX.")
        print("   Usa el dashboard para registrar evaluaciones durante tus sesiones.")
