"""
ATOM - Modelo Predictivo de Fatiga v2.0
========================================
Modelo personal (Random Forest) que aprende TUS patrones de fatiga a partir de
etiquetas KSS (escala de somnolencia de Karolinska, 1-9) que Atomica pregunta
por voz durante la jornada.

Cada etiqueta KSS guarda una "foto" de las metricas de los 10 minutos previos
(`caracteristicas`). El modelo se entrena con esas fotos, asi:
  - cada respuesta es UNA muestra (sin inflar el dataset con minutos vecinos)
  - las etiquetas sobreviven al borrado de datos crudos (14 dias)
  - entrenamiento y prediccion calculan las metricas con la MISMA funcion

Mientras no haya suficientes etiquetas se usa una heuristica relativa al EAR
habitual del usuario.

Changelog:
  - 2026-04-07: Implementacion inicial con Random Forest + NASA-TLX
  - 2026-11-13: v2.0 etiquetas KSS con foto de metricas; validacion por etiqueta
"""

import os
import pickle
import sqlite3
import statistics
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_DIR, "..", "data", "atom_telemetry.db")
MODEL_PATH = os.path.join(_DIR, "..", "data", "fatigue_model_kss.pkl")

DEFAULT_EAR_THRESHOLD = 0.22
INTERVALO_S = 10            # los sensores guardan una fila cada 10 s
VENTANA_MIN = 10            # minutos de metricas previos a cada etiqueta
MIN_CAMARA_MIN = 3          # minimo de camara en la ventana para que sirva
PAUSA_MIN = 5               # minutos sin actividad que cortan una sesion

KSS_FATIGA = 7              # KSS >= 7 ("somnoliento") = fatigado (corte usual en la literatura)
MIN_ETIQUETAS = 20          # minimo para entrenar
MIN_POR_CLASE = 5           # minimo de respuestas fatigado y no fatigado

FEATURES = ["ear_ratio", "ear_std", "parpadeos_min", "teclas_min", "clics_min",
            "pitch", "min_sesion", "hora"]


def _fmt(d):
    return d.strftime("%Y-%m-%d %H:%M:%S")


def ear_personal(conn, dias=7, min_filas=30):
    """Mediana del EAR real de los ultimos dias (o None si hay pocos datos)."""
    desde = _fmt(datetime.now() - timedelta(days=dias))
    ears = [r[0] for r in conn.execute(
        "SELECT avg_ear FROM visual_telemetry WHERE source = 'real' AND avg_ear IS NOT NULL "
        "AND timestamp >= ?", (desde,))]
    return statistics.median(ears) if len(ears) >= min_filas else None


def caracteristicas(conn, fin=None, minutos=VENTANA_MIN, ear_base=None):
    """Metricas de los `minutos` previos a `fin`. None si hay poca camara."""
    fin = fin or datetime.now()
    ini = fin - timedelta(minutes=minutos)
    vis = conn.execute(
        "SELECT avg_ear, blinks, pitch FROM visual_telemetry WHERE source = 'real' "
        "AND timestamp > ? AND timestamp <= ?", (_fmt(ini), _fmt(fin))).fetchall()
    ears = [v[0] for v in vis if v[0] is not None]
    min_cam = len(vis) * INTERVALO_S / 60
    if min_cam < MIN_CAMARA_MIN or len(ears) < 6:
        return None
    ear_base = ear_base or ear_personal(conn)
    ear = statistics.fmean(ears)

    sis = conn.execute(
        "SELECT SUM(keystrokes), SUM(mouse_clicks), COUNT(*) FROM system_telemetry "
        "WHERE timestamp > ? AND timestamp <= ?", (_fmt(ini), _fmt(fin))).fetchone()
    min_sis = max((sis[2] or 0) * INTERVALO_S / 60, 1 / 6)

    # Minutos de trabajo continuo hasta `fin` (se corta en una pausa de PAUSA_MIN)
    actividad = [datetime.strptime(t, "%Y-%m-%d %H:%M:%S") for (t,) in conn.execute(
        "SELECT timestamp FROM system_telemetry WHERE timestamp <= ? AND timestamp >= ? "
        "AND keystrokes + mouse_clicks > 0 ORDER BY timestamp DESC",
        (_fmt(fin), _fmt(fin - timedelta(hours=8))))]
    inicio = previo = fin
    for t in actividad:
        if (previo - t).total_seconds() > PAUSA_MIN * 60:
            break
        inicio = previo = t

    return {
        "ear": round(ear, 4),
        "ear_ratio": round(ear / ear_base, 4) if ear_base else None,
        "ear_std": round(statistics.pstdev(ears), 4),
        "parpadeos_min": round(sum(v[1] or 0 for v in vis) / min_cam, 2),
        "teclas_min": round((sis[0] or 0) / min_sis, 2),
        "clics_min": round((sis[1] or 0) / min_sis, 2),
        "pitch": round(statistics.fmean(v[2] or 0 for v in vis), 2),
        "min_sesion": round((fin - inicio).total_seconds() / 60, 1),
        "hora": round(fin.hour + fin.minute / 60, 2),
        "min_camara": round(min_cam, 1),
    }


class FatiguePredictor:
    """
    Prediccion de fatiga:
      - Con modelo entrenado: Random Forest sobre `caracteristicas` actuales.
      - Sin modelo: heuristica relativa al EAR habitual del usuario.
    """

    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.model = None
        self.info = {}
        self.is_trained = False
        # Se conservan por compatibilidad con el dashboard / agent_trigger
        self.baseline_ear = None
        self.baseline_apm = None
        self.personal_ear = None
        self.compute_personal_ear()
        self._load_model()

    def compute_personal_ear(self, days=7, min_rows=30):
        try:
            with sqlite3.connect(self.db_path) as conn:
                self.personal_ear = ear_personal(conn, days, min_rows)
            if self.personal_ear:
                print(f"[ML] EAR personal (datos reales): {self.personal_ear:.3f}")
        except Exception as e:
            print(f"[ML] No se pudo calcular EAR personal: {e}")
        return self.personal_ear

    def _load_model(self):
        if not os.path.exists(MODEL_PATH):
            print("[ML] Sin modelo personal todavia: heuristica relativa al EAR habitual.")
            return
        try:
            with open(MODEL_PATH, "rb") as f:
                datos = pickle.load(f)
            self.model, self.info = datos["modelo"], datos["info"]
            self.is_trained = True
            print(f"[ML] Modelo personal cargado ({self.info.get('etiquetas')} etiquetas KSS, "
                  f"F1 validacion {self.info.get('f1', float('nan')):.2f}).")
        except Exception as e:
            print(f"[ML] Error cargando modelo: {e}. Usando heuristica.")

    # ── Entrenamiento ────────────────────────────────────────────
    def _prepare_training_data(self):
        with sqlite3.connect(self.db_path) as conn:
            try:
                df = pd.read_sql_query(
                    f"SELECT valor, {', '.join(FEATURES)} FROM kss WHERE ear_ratio IS NOT NULL", conn)
            except Exception:
                return None, None
        if df.empty:
            return None, None
        X = df[FEATURES].fillna(0)
        y = (df["valor"] >= KSS_FATIGA).astype(int)
        return X, y

    def estado_etiquetas(self):
        """Cuantas etiquetas hay y cuantas faltan para entrenar."""
        X, y = self._prepare_training_data()
        n = 0 if X is None else len(X)
        fat = 0 if y is None else int(y.sum())
        return {"etiquetas": n, "fatigado": fat, "no_fatigado": n - fat,
                "faltan": max(0, MIN_ETIQUETAS - n),
                "faltan_fatigado": max(0, MIN_POR_CLASE - fat),
                "faltan_no_fatigado": max(0, MIN_POR_CLASE - (n - fat))}

    def train_model(self):
        """Entrena con las etiquetas KSS. Retorna el F1 de validacion o None."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import StratifiedKFold, cross_val_score

        print("\n" + "=" * 50)
        print("  ATOM ML - Entrenando modelo personal de fatiga")
        print("=" * 50)
        X, y = self._prepare_training_data()
        e = self.estado_etiquetas()
        if X is None or e["faltan"] or e["faltan_fatigado"] or e["faltan_no_fatigado"]:
            print(f"[ML] Aun no: {e['etiquetas']} etiquetas KSS con camara "
                  f"({e['fatigado']} fatigado / {e['no_fatigado']} no). Se necesitan "
                  f"{MIN_ETIQUETAS} y al menos {MIN_POR_CLASE} de cada tipo.")
            return None

        modelo = RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=2,
                                        class_weight="balanced", random_state=42)
        # Una muestra por etiqueta: la validacion no mezcla minutos de la misma respuesta
        k = min(5, int(y.sum()), int(len(y) - y.sum()))
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        f1 = cross_val_score(modelo, X, y, cv=cv, scoring="f1")
        print(f"[ML] Validacion cruzada (k={k}): F1 = {f1.mean():.3f} ± {f1.std():.3f}")

        modelo.fit(X, y)
        imp = sorted(zip(FEATURES, modelo.feature_importances_), key=lambda x: -x[1])
        print("[ML] Importancia de cada metrica:")
        for nombre, v in imp:
            print(f"  {nombre:14s} {v:.3f} {'█' * int(v * 40)}")

        self.model, self.is_trained = modelo, True
        self.info = {"etiquetas": len(y), "fatigado": int(y.sum()), "f1": float(f1.mean()),
                     "entrenado": _fmt(datetime.now()),
                     "importancia": {n: round(float(v), 3) for n, v in imp}}
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({"modelo": modelo, "info": self.info}, f)
        print("[ML] Modelo guardado.")
        return float(f1.mean())

    # ── Prediccion ───────────────────────────────────────────────
    def predict(self, metrics_dict):
        """Probabilidad de fatiga (0-1). Con modelo usa los ultimos 10 min de la
        base; sin modelo (o con poca camara) usa la heuristica sobre metrics_dict."""
        if self.is_trained:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    f = caracteristicas(conn, ear_base=self.personal_ear)
                if f and f["ear_ratio"] is not None:
                    X = pd.DataFrame([{k: f[k] for k in FEATURES}]).fillna(0)
                    return float(self.model.predict_proba(X)[0][1])
            except Exception as e:
                print(f"[ML] Error prediciendo con el modelo: {e}")
        return self._heuristic_prediction(metrics_dict)

    def _heuristic_prediction(self, metrics):
        """Probabilidad aproximada por umbrales (mientras no hay modelo)."""
        score = 0.0

        ear = metrics.get('avg_ear', 0.25)
        if ear is None or ear != ear:  # NULL/NaN: intervalo sin EAR valido
            ear = self.personal_ear or 0.25
        if self.personal_ear:
            # Relativo al EAR normal del usuario: cada persona abre
            # los ojos distinto, un umbral fijo da falsos positivos.
            ratio = ear / self.personal_ear
            if ratio < 0.80:
                score += 0.4
            elif ratio < 0.88:
                score += 0.25
            elif ratio < 0.94:
                score += 0.1
        elif ear < 0.20:
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
        elif blinks < 5:  # Fijacion prolongada
            score += 0.1

        return min(score, 1.0)

    def get_adaptive_threshold(self):
        """Umbral EAR personal: percentil 15 del EAR real reciente."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                ears = [r[0] for r in conn.execute(
                    "SELECT avg_ear FROM visual_telemetry WHERE source = 'real' "
                    "AND avg_ear IS NOT NULL ORDER BY id DESC LIMIT 1000")]
            if len(ears) > 50:
                return float(np.percentile(ears, 15))
        except Exception:
            pass
        return DEFAULT_EAR_THRESHOLD

    def get_status_report(self):
        """Resumen del estado del modelo para el dashboard."""
        return {
            'mode': (f"ML personal ({self.info.get('etiquetas')} etiquetas KSS)"
                     if self.is_trained else 'Heuristico (EAR relativo)'),
            'baseline_ear': self.personal_ear,
            'baseline_apm': None,
            'threshold': self.get_adaptive_threshold(),
        }


# ─── Ejecucion directa: entrenar modelo ───────────────────────────────
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    p = FatiguePredictor()
    print(p.estado_etiquetas())
    resultado = p.train_model()
    if resultado is not None:
        print(f"\nModelo listo. F1 de validacion: {resultado:.3f}")
