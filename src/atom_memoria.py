"""
ATOM - Memoria de largo plazo
=============================
Como la memoria de un asistente de IA: no se guarda todo, solo lo relevante.

  Corto plazo  -> telemetria cruda (fila cada 10 s). Se conserva RETENER_DIAS
                  dias; despues se resume y se borra.
  Episodica    -> tabla `eventos`: solo hechos importantes (alertas, sesiones
                  largas sin pausa, mala postura sostenida, NASA-TLX).
  Resumenes    -> tabla `resumen_diario`: una fila por dia, permanente.
  Semantica    -> tabla `memoria`: preferencias, metas y patrones del usuario.
                  Claude decide que guardar (herramientas recordar / olvidar).
                  Lo poco importante que no se usa en OLVIDO_DIAS se olvida.

Uso:
    import atom_memoria as mem
    mem.consolidar()                          # al arrancar ATOM
    mem.registrar_evento("alerta_fatiga", "prob 78%, EAR 0.19")
    mem.recordar("Prefiere descansos cada 50 minutos", "preferencia", 4)
"""

import difflib
import json
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "atom_telemetry.db")

RETENER_DIAS = 14        # datos crudos que se conservan
OLVIDO_DIAS = 60         # memoria poco importante sin uso -> se olvida
MAX_MEMORIAS = 100       # tope de memoria semantica
INTERVALO_S = 10         # los sensores guardan una fila cada 10 s
PAUSA_MIN = 5            # minutos sin actividad que cuentan como pausa
SESION_LARGA_MIN = 90    # sesion continua que merece un evento
POSTURA_LARGA_MIN = 10   # mala postura sostenida que merece un evento

CATEGORIAS = ("preferencia", "dato_personal", "meta", "patron", "salud")

# Caracteristicas que se congelan con cada respuesta KSS (ver fatigue_model.caracteristicas)
COLUMNAS_KSS = ("ear", "ear_ratio", "ear_std", "parpadeos_min", "teclas_min", "clics_min",
                "pitch", "min_sesion", "hora", "min_camara")


def nombre_app(titulo, app=None):
    """Solo el nombre del programa, nunca el titulo completo (revela busquedas,
    documentos, tareas). `app` viene del sensor desde nov-2026; para filas
    anteriores se deduce del titulo."""
    if app:
        return app
    t = (titulo or "").strip()
    if not t:
        return ""
    # La terminal de Claude Code pone un icono animado al inicio del titulo
    if t[0] in "✳✶✻✽✢·◐◑◒◓⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏":
        return "Claude Code"
    t = re.sub(r"^[^\w(\[¿¡]+", "", t)
    return t.rsplit(" - ", 1)[-1].strip()


@contextmanager
def _db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _crear_tablas(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS resumen_diario (
            fecha TEXT PRIMARY KEY,
            inicio TEXT, fin TEXT,
            min_activos INTEGER, teclas INTEGER, clics INTEGER,
            ventanas TEXT,
            min_camara REAL, ear REAL, parpadeos_min REAL, pct_mala_postura REAL,
            sesion_max_min INTEGER,
            alertas_fatiga INTEGER, alertas_postura INTEGER,
            tlx_compuesto REAL
        );
        CREATE TABLE IF NOT EXISTS eventos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            tipo TEXT NOT NULL,
            detalle TEXT,
            derivado INTEGER DEFAULT 0      -- 1 = calculado al consolidar
        );
        CREATE INDEX IF NOT EXISTS idx_eventos_ts ON eventos(timestamp);
        CREATE TABLE IF NOT EXISTS memoria (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            contenido TEXT NOT NULL,
            categoria TEXT,
            importancia INTEGER DEFAULT 3,  -- 1 trivial .. 5 esencial
            creado TEXT, actualizado TEXT, ultimo_uso TEXT,
            usos INTEGER DEFAULT 0
        );
        -- Escala de somnolencia de Karolinska (1 muy alerta .. 9 muy somnoliento)
        -- con la foto de las metricas previas: es la etiqueta del modelo personal
        -- y sobrevive al borrado de datos crudos.
        CREATE TABLE IF NOT EXISTS kss (
            timestamp TEXT PRIMARY KEY,
            valor INTEGER NOT NULL,
            origen TEXT,
            ear REAL, ear_ratio REAL, ear_std REAL, parpadeos_min REAL,
            teclas_min REAL, clics_min REAL, pitch REAL, min_sesion REAL,
            hora REAL, min_camara REAL
        );
    """)
    columnas = {r[1] for r in conn.execute("PRAGMA table_info(resumen_diario)")}
    if "kss" not in columnas:
        conn.execute("ALTER TABLE resumen_diario ADD COLUMN kss REAL")


# ══════════════════════════════════════════════════════════════════
#  ETIQUETAS DE FATIGA (KSS)
# ══════════════════════════════════════════════════════════════════

def registrar_kss(valor, origen="pregunta"):
    """Guarda la respuesta KSS junto con las metricas de los 10 min previos."""
    from fatigue_model import caracteristicas
    valor = max(1, min(9, int(round(float(valor)))))
    ahora = datetime.now()
    ts = ahora.strftime("%Y-%m-%d %H:%M:%S")
    with _db() as c:
        _crear_tablas(c)
        feats = caracteristicas(c, ahora) or {}
        cols = ["timestamp", "valor", "origen"] + [k for k in COLUMNAS_KSS if k in feats]
        vals = [ts, valor, origen] + [feats[k] for k in COLUMNAS_KSS if k in feats]
        c.execute(f"INSERT OR REPLACE INTO kss ({', '.join(cols)}) VALUES "
                  f"({', '.join('?' * len(cols))})", vals)
        total = c.execute("SELECT COUNT(*) FROM kss WHERE ear_ratio IS NOT NULL").fetchone()[0]
    registrar_evento("kss", f"{valor}/9", ts)
    return {"guardado": True, "valor": valor, "con_datos_de_camara": bool(feats),
            "etiquetas_utiles": total}


def kss_hoy():
    hoy = datetime.now().strftime("%Y-%m-%d")
    with _db() as c:
        _crear_tablas(c)
        filas = c.execute("SELECT timestamp, valor FROM kss WHERE timestamp LIKE ? "
                          "ORDER BY timestamp", (hoy + "%",)).fetchall()
        utiles = c.execute("SELECT COUNT(*) FROM kss WHERE ear_ratio IS NOT NULL").fetchone()[0]
    return filas, utiles


# ══════════════════════════════════════════════════════════════════
#  MEMORIA EPISODICA
# ══════════════════════════════════════════════════════════════════

def registrar_evento(tipo, detalle="", timestamp=None, derivado=False):
    ts = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _db() as c:
        _crear_tablas(c)
        c.execute("INSERT INTO eventos (timestamp, tipo, detalle, derivado) VALUES (?,?,?,?)",
                  (ts, tipo, detalle, int(derivado)))


def eventos_recientes(dias=7, limite=15):
    desde = (datetime.now() - timedelta(days=dias)).strftime("%Y-%m-%d")
    with _db() as c:
        _crear_tablas(c)
        filas = c.execute("SELECT timestamp, tipo, detalle FROM eventos WHERE timestamp >= ? "
                          "ORDER BY timestamp DESC LIMIT ?", (desde, limite)).fetchall()
    return [f"{t[5:16]} {tipo}" + (f": {d}" if d else "") for t, tipo, d in filas]


# ══════════════════════════════════════════════════════════════════
#  CONSOLIDACION (corto plazo -> resumen + eventos)
# ══════════════════════════════════════════════════════════════════

def _rachas(timestamps, condicion, hueco_max_s):
    """Duraciones (min) e inicio de rachas consecutivas donde condicion es True."""
    rachas, inicio, previo = [], None, None
    for ts, ok in zip(timestamps, condicion):
        t = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        if ok and inicio is not None and (t - previo).total_seconds() <= hueco_max_s:
            previo = t
            continue
        if inicio is not None:
            rachas.append((inicio, (previo - inicio).total_seconds() / 60))
            inicio = None
        if ok:
            inicio = previo = t
    if inicio is not None:
        rachas.append((inicio, (previo - inicio).total_seconds() / 60))
    return rachas


def _resumir_dia(c, fecha):
    patron = fecha + "%"
    tiene_app = "app" in {r[1] for r in c.execute("PRAGMA table_info(system_telemetry)")}
    sis = c.execute("SELECT timestamp, keystrokes, mouse_clicks, active_window, "
                    f"{'app' if tiene_app else 'NULL'} FROM system_telemetry "
                    "WHERE timestamp LIKE ? ORDER BY timestamp", (patron,)).fetchall()
    vis = c.execute("SELECT timestamp, avg_ear, blinks, pitch, yaw FROM visual_telemetry "
                    "WHERE source = 'real' AND timestamp LIKE ? ORDER BY timestamp", (patron,)).fetchall()
    if not sis and not vis:
        return None

    r = {"fecha": fecha}
    if sis:
        activos = [s for s in sis if (s[1] or 0) + (s[2] or 0) > 0]
        conteo = {}
        for s in sis:
            # Solo la aplicacion: el titulo revela busquedas/documentos y gasta tokens
            app = nombre_app(s[3], s[4])
            if app:
                conteo[app] = conteo.get(app, 0) + 1
        sesiones = _rachas([s[0] for s in sis], [(s[1] or 0) + (s[2] or 0) > 0 for s in sis],
                           PAUSA_MIN * 60)
        r.update(inicio=sis[0][0][11:16], fin=sis[-1][0][11:16],
                 min_activos=round(len(activos) * INTERVALO_S / 60),
                 teclas=sum(s[1] or 0 for s in sis), clics=sum(s[2] or 0 for s in sis),
                 ventanas=json.dumps([v for v, _ in sorted(conteo.items(), key=lambda x: -x[1])[:3]],
                                     ensure_ascii=False),
                 sesion_max_min=round(max((d for _, d in sesiones), default=0)))
        for ini, dur in sesiones:
            if dur >= SESION_LARGA_MIN:
                c.execute("INSERT INTO eventos (timestamp, tipo, detalle, derivado) VALUES (?,?,?,1)",
                          (ini.strftime("%Y-%m-%d %H:%M:%S"), "sesion_larga",
                           f"{round(dur)} min sin pausa"))
    if vis:
        ears = [v[1] for v in vis if v[1] is not None]
        mala = [abs(v[3] or 0) > 25 or abs(v[4] or 0) > 30 for v in vis]
        min_cam = len(vis) * INTERVALO_S / 60
        r.update(min_camara=round(min_cam, 1),
                 ear=round(sum(ears) / len(ears), 3) if ears else None,
                 parpadeos_min=round(sum(v[2] or 0 for v in vis) / max(min_cam, 1 / 6), 1),
                 pct_mala_postura=round(100 * sum(mala) / len(vis)))
        for ini, dur in _rachas([v[0] for v in vis], mala, 3 * INTERVALO_S):
            if dur >= POSTURA_LARGA_MIN:
                c.execute("INSERT INTO eventos (timestamp, tipo, detalle, derivado) VALUES (?,?,?,1)",
                          (ini.strftime("%Y-%m-%d %H:%M:%S"), "mala_postura_sostenida",
                           f"{round(dur)} min"))

    alertas = dict(c.execute("SELECT tipo, COUNT(*) FROM eventos WHERE timestamp LIKE ? "
                             "AND derivado = 0 GROUP BY tipo", (patron,)).fetchall())
    r["alertas_fatiga"] = alertas.get("alerta_fatiga", 0)
    r["alertas_postura"] = alertas.get("alerta_postura", 0)
    tlx = c.execute("SELECT AVG((mental+fisica+temporal+rendimiento+esfuerzo+frustracion)/6.0) "
                    "FROM nasa_tlx WHERE timestamp LIKE ?", (patron,)).fetchone()[0]
    r["tlx_compuesto"] = round(tlx, 1) if tlx is not None else None
    kss = c.execute("SELECT AVG(valor) FROM kss WHERE timestamp LIKE ?", (patron,)).fetchone()[0]
    r["kss"] = round(kss, 1) if kss is not None else None
    return r


def consolidar(retener_dias=RETENER_DIAS, verbose=True):
    """
    Resume los dias anteriores a hoy, guarda eventos relevantes y borra los
    datos crudos con mas de `retener_dias`. Es idempotente: se puede llamar
    en cada arranque.
    """
    hoy = datetime.now().strftime("%Y-%m-%d")
    corte = (datetime.now() - timedelta(days=retener_dias)).strftime("%Y-%m-%d")
    with _db() as c:
        _crear_tablas(c)
        dias = sorted({d for (d,) in c.execute(
            "SELECT DISTINCT substr(timestamp,1,10) FROM system_telemetry WHERE timestamp < ? "
            "UNION SELECT DISTINCT substr(timestamp,1,10) FROM visual_telemetry "
            "WHERE source = 'real' AND timestamp < ?", (hoy, hoy))})
        resumidos = 0
        for fecha in dias:
            # Recalcular solo mientras haya datos crudos; los eventos derivados
            # del dia se regeneran para no duplicarlos.
            c.execute("DELETE FROM eventos WHERE derivado = 1 AND timestamp LIKE ?", (fecha + "%",))
            r = _resumir_dia(c, fecha)
            if r:
                cols = ", ".join(r)
                c.execute(f"INSERT OR REPLACE INTO resumen_diario ({cols}) VALUES "
                          f"({', '.join('?' * len(r))})", list(r.values()))
                resumidos += 1

        # Borrar crudos viejos (solo dias que ya tienen resumen)
        con_resumen = "substr(timestamp,1,10) IN (SELECT fecha FROM resumen_diario)"
        b_sis = c.execute(f"DELETE FROM system_telemetry WHERE timestamp < ? AND {con_resumen}",
                          (corte,)).rowcount
        b_vis = c.execute(f"DELETE FROM visual_telemetry WHERE timestamp < ? AND "
                          f"(source IS NULL OR source != 'real' OR {con_resumen})", (corte,)).rowcount

        # Olvido: memoria poco importante que no se ha usado en mucho tiempo
        limite_olvido = (datetime.now() - timedelta(days=OLVIDO_DIAS)).strftime("%Y-%m-%d")
        olvidadas = c.execute("DELETE FROM memoria WHERE importancia <= 2 AND "
                              "COALESCE(ultimo_uso, creado) < ?", (limite_olvido,)).rowcount

    if b_sis + b_vis > 1000:
        with _db() as c:
            c.execute("VACUUM")
    if verbose:
        print(f"[Memoria] {resumidos} dias resumidos | crudos borrados: {b_sis} sistema, "
              f"{b_vis} visual (> {retener_dias} dias) | memorias olvidadas: {olvidadas}")
    return {"dias_resumidos": resumidos, "borrados": b_sis + b_vis, "olvidadas": olvidadas}


def historial(dias=7):
    """Resumenes compactos de los ultimos dias (para Claude)."""
    desde = (datetime.now() - timedelta(days=dias)).strftime("%Y-%m-%d")
    with _db() as c:
        _crear_tablas(c)
        c.row_factory = sqlite3.Row
        filas = c.execute("SELECT * FROM resumen_diario WHERE fecha >= ? ORDER BY fecha",
                          (desde,)).fetchall()
    # Solo campos con valor: menos tokens
    return [{k: f[k] for k in f.keys() if f[k] not in (None, 0, "[]")} for f in filas]


# ══════════════════════════════════════════════════════════════════
#  MEMORIA SEMANTICA
# ══════════════════════════════════════════════════════════════════

def _similar(a, b):
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def recordar(contenido, categoria="preferencia", importancia=3):
    contenido = " ".join(str(contenido).split())[:200]
    categoria = categoria if categoria in CATEGORIAS else "preferencia"
    importancia = max(1, min(5, int(importancia)))
    ahora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _db() as c:
        _crear_tablas(c)
        existentes = c.execute("SELECT id, contenido FROM memoria").fetchall()
        parecido = max(existentes, key=lambda m: _similar(m[1], contenido), default=None)
        if parecido and _similar(parecido[1], contenido) >= 0.75:
            # Ya existe algo casi igual: actualizar en vez de duplicar
            c.execute("UPDATE memoria SET contenido=?, categoria=?, importancia=MAX(importancia, ?), "
                      "actualizado=? WHERE id=?", (contenido, categoria, importancia, ahora, parecido[0]))
            accion = "actualizada"
        else:
            c.execute("INSERT INTO memoria (contenido, categoria, importancia, creado, actualizado) "
                      "VALUES (?,?,?,?,?)", (contenido, categoria, importancia, ahora, ahora))
            accion = "guardada"
        # Tope: si hay demasiadas, se olvida la menos importante y menos usada
        exceso = c.execute("SELECT COUNT(*) FROM memoria").fetchone()[0] - MAX_MEMORIAS
        if exceso > 0:
            c.execute("DELETE FROM memoria WHERE id IN (SELECT id FROM memoria ORDER BY "
                      "importancia, usos, COALESCE(ultimo_uso, creado) LIMIT ?)", (exceso,))
    return {"ok": True, "memoria": accion}


def olvidar(contenido):
    with _db() as c:
        _crear_tablas(c)
        existentes = c.execute("SELECT id, contenido FROM memoria").fetchall()
        mejor = max(existentes, key=lambda m: _similar(m[1], contenido), default=None)
        if not mejor or _similar(mejor[1], contenido) < 0.45:
            return {"ok": False, "error": "No encontre ese recuerdo"}
        c.execute("DELETE FROM memoria WHERE id = ?", (mejor[0],))
    return {"ok": True, "olvidado": mejor[1]}


def memorias_para_prompt(limite=15):
    """Las memorias mas relevantes, una por linea, para el system prompt."""
    with _db() as c:
        _crear_tablas(c)
        filas = c.execute("SELECT id, contenido FROM memoria ORDER BY importancia DESC, "
                          "COALESCE(ultimo_uso, creado) DESC LIMIT ?", (limite,)).fetchall()
        if filas:
            ahora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            c.executemany("UPDATE memoria SET usos = usos + 1, ultimo_uso = ? WHERE id = ?",
                          [(ahora, i) for i, _ in filas])
    return "\n".join(f"- {t}" for _, t in filas)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    consolidar()
    print("Historial:", json.dumps(historial(30), ensure_ascii=False))
    print("Memorias:\n" + (memorias_para_prompt() or "(vacia)"))
