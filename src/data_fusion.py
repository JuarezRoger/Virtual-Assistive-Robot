"""
ATOM - Fusión de Datos Multimodal v2.0
=======================================
Fusiona las tres fuentes de telemetría (sistema, visual, NASA-TLX)
en un dataset unificado con métricas derivadas para análisis y ML.

Mejoras v2.0:
  - Integra blinks, pitch, yaw, roll del sensor visual mejorado
  - Calcula métricas derivadas: ear_variability, blink_rate, posture_score
  - Genera gráfica multi-panel profesional
  - Exporta CSV fusionado para análisis externo

Changelog:
  - 2026-04-07: Integración de postura HPE, métricas derivadas, multi-panel
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np


def fuse_and_visualize(db_path="../data/atom_telemetry.db", export_csv=True):
    """
    Lee, fusiona y visualiza los datos de todos los sensores ATOM.

    Args:
        db_path: Ruta a la base de datos SQLite
        export_csv: Si True, exporta el dataset fusionado a CSV
    """
    print("=" * 55)
    print("  ATOM - Fusión de Datos Multimodal v2.0")
    print("=" * 55)

    conn = sqlite3.connect(db_path)

    # 1. Leer ambas tablas
    try:
        df_sys = pd.read_sql_query("SELECT * FROM system_telemetry", conn)
        df_vis = pd.read_sql_query("SELECT * FROM visual_telemetry", conn)
    except Exception as e:
        print(f"Error leyendo la base de datos: {e}")
        conn.close()
        return None

    # Intentar leer NASA-TLX (opcional)
    try:
        df_tlx = pd.read_sql_query("SELECT * FROM nasa_tlx", conn)
    except Exception:
        df_tlx = pd.DataFrame()

    conn.close()

    if df_sys.empty or df_vis.empty:
        print("Datos insuficientes. Deja que los sensores corran un poco más.")
        return None

    print(f"  Sistema: {len(df_sys)} registros")
    print(f"  Visual:  {len(df_vis)} registros")
    print(f"  TLX:     {len(df_tlx)} registros")

    # 2. Convertir timestamps y establecer como índice
    df_sys['timestamp'] = pd.to_datetime(df_sys['timestamp'])
    df_vis['timestamp'] = pd.to_datetime(df_vis['timestamp'])

    df_sys.set_index('timestamp', inplace=True)
    df_vis.set_index('timestamp', inplace=True)

    print("\nFusionando líneas temporales (bloques de 1 minuto)...")

    # 3. Remuestrear a bloques de 1 minuto
    sys_resampled = df_sys.resample('1min').agg({
        'keystrokes': 'sum',
        'mouse_clicks': 'sum',
        'cpu_usage': 'mean',
        'active_window': lambda x: x.mode()[0] if not x.empty else "Desconocida"
    }).dropna()

    # Construir agregaciones visuales (compatible con esquema viejo y nuevo)
    vis_agg = {'avg_ear': 'mean'}

    if 'blinks' in df_vis.columns:
        vis_agg['blinks'] = 'sum'
    if 'pitch' in df_vis.columns:
        vis_agg['pitch'] = 'mean'
    if 'yaw' in df_vis.columns:
        vis_agg['yaw'] = 'mean'
    if 'roll' in df_vis.columns:
        vis_agg['roll'] = 'mean'

    vis_resampled = df_vis.resample('1min').agg(vis_agg).dropna()

    # Agregar variabilidad de EAR (métrica derivada)
    vis_resampled['ear_std'] = df_vis['avg_ear'].resample('1min').std().fillna(0)

    # 4. Fusión interna
    df_fusion = pd.merge(
        sys_resampled, vis_resampled,
        left_index=True, right_index=True
    )

    if df_fusion.empty:
        print("Aún no hay suficientes datos superpuestos para fusionar.")
        return None

    # 5. Métricas derivadas
    if 'blinks' in df_fusion.columns:
        df_fusion['blink_rate'] = df_fusion['blinks']  # Ya es por minuto
    if 'pitch' in df_fusion.columns and 'yaw' in df_fusion.columns:
        df_fusion['posture_score'] = np.sqrt(
            df_fusion['pitch']**2 + df_fusion['yaw']**2
        )

    # Índice de carga cognitiva compuesto (normalizado 0-1)
    ear_norm = 1 - (df_fusion['avg_ear'] - 0.15) / (0.35 - 0.15)  # Invertido
    apm_norm = 1 - df_fusion['keystrokes'] / df_fusion['keystrokes'].max() if df_fusion['keystrokes'].max() > 0 else 0
    df_fusion['cognitive_load_index'] = (ear_norm.clip(0, 1) * 0.5 + apm_norm * 0.5).clip(0, 1)

    print(f"\n--- Dataset Fusionado: {len(df_fusion)} registros ---")
    cols_to_show = ['keystrokes', 'avg_ear']
    if 'blinks' in df_fusion.columns:
        cols_to_show.append('blinks')
    if 'pitch' in df_fusion.columns:
        cols_to_show.append('pitch')
    cols_to_show.append('cognitive_load_index')
    print(df_fusion[cols_to_show].tail(10))

    # 6. Exportar CSV
    if export_csv:
        csv_path = db_path.replace('.db', '_fused.csv')
        df_fusion.to_csv(csv_path)
        print(f"\n  CSV exportado: {csv_path}")

    # 7. Visualización
    _plot_multipanel(df_fusion)

    return df_fusion


def _plot_multipanel(df):
    """Genera gráfica multi-panel profesional con todas las métricas."""
    sns.set_theme(style="darkgrid")

    has_posture = 'pitch' in df.columns and df['pitch'].abs().sum() > 0
    has_blinks = 'blinks' in df.columns and df['blinks'].sum() > 0

    n_panels = 2 + int(has_posture) + int(has_blinks)
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 4 * n_panels),
                              sharex=True, facecolor='#1a1a2e')

    for ax in axes:
        ax.set_facecolor('#16213e')

    panel = 0

    # Panel 1: EAR + Umbral
    ax = axes[panel]
    ax.plot(df.index, df['avg_ear'], color='#00d4ff', linewidth=2, label='EAR Promedio')
    ax.fill_between(df.index, df['avg_ear'] - df.get('ear_std', 0),
                     df['avg_ear'] + df.get('ear_std', 0),
                     alpha=0.2, color='#00d4ff')
    ax.axhline(y=0.22, color='#ff4444', linestyle='--', linewidth=1.5,
               label='Umbral Fijo (0.22)', alpha=0.7)
    ax.set_ylabel('Eye Aspect Ratio', color='white', fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title('ATOM: Fusión Sensorial Multimodal v2.0',
                  fontsize=16, color='white', pad=15, fontweight='bold')
    panel += 1

    # Panel 2: Actividad Motora (APM)
    ax = axes[panel]
    ax.bar(df.index, df['keystrokes'], width=0.0005, color='#4ecdc4',
           alpha=0.8, label='Teclado')
    ax.bar(df.index, df['mouse_clicks'], width=0.0005, color='#ff6b6b',
           alpha=0.6, label='Clics', bottom=df['keystrokes'])
    ax.set_ylabel('Actividad (por minuto)', color='white', fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    panel += 1

    # Panel 3: Parpadeos (si disponible)
    if has_blinks:
        ax = axes[panel]
        ax.plot(df.index, df['blinks'], color='#feca57', linewidth=2,
                marker='o', markersize=3, label='Parpadeos/min')
        ax.axhspan(15, 20, alpha=0.1, color='green', label='Rango normal')
        ax.set_ylabel('Blinks / min', color='white', fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        panel += 1

    # Panel 4: Postura (si disponible)
    if has_posture:
        ax = axes[panel]
        ax.plot(df.index, df['pitch'], color='#ff9ff3', linewidth=2, label='Pitch (frontal)')
        ax.plot(df.index, df['yaw'], color='#54a0ff', linewidth=2, label='Yaw (lateral)')
        ax.axhline(y=25, color='red', linestyle=':', alpha=0.5)
        ax.axhline(y=-25, color='red', linestyle=':', alpha=0.5)
        ax.set_ylabel('Ángulo (grados)', color='white', fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)

    # Formato de eje X
    axes[-1].set_xlabel('Tiempo de Sesión', color='white', fontweight='bold')
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    for ax in axes:
        ax.tick_params(colors='white')

    fig.tight_layout()
    plt.savefig('../data/atom_fusion_report.png', dpi=150, bbox_inches='tight',
                facecolor='#1a1a2e')
    print("  Gráfica guardada: ../data/atom_fusion_report.png")
    plt.show()


if __name__ == "__main__":
    fuse_and_visualize()
