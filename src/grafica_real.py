import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

print("📊 Extrayendo telemetría real de ATOM...")

db_path = "../data/atom_telemetry.db"

try:
    # 1. Conectar y extraer datos
    conn = sqlite3.connect(db_path)
    df_vis = pd.read_sql_query("SELECT timestamp, avg_ear FROM visual_telemetry", conn)
    df_sys = pd.read_sql_query("SELECT timestamp, keystrokes FROM system_telemetry", conn)
    conn.close()

    if df_vis.empty or df_sys.empty:
        print("⚠️ La base de datos aún tiene muy pocos datos. Deja correr a ATOM un rato más.")
        exit()

    # 2. Convertir las fechas para que Matplotlib las entienda
    df_vis['timestamp'] = pd.to_datetime(df_vis['timestamp'])
    df_sys['timestamp'] = pd.to_datetime(df_sys['timestamp'])

    # 3. Configurar la gráfica estilo académico
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # --- PANEL SUPERIOR: EAR REAL ---
    ax1.plot(df_vis['timestamp'], df_vis['avg_ear'], color='#2c3e50', linewidth=1.5, label='EAR Capturado')
    ax1.axhline(y=0.22, color='#e74c3c', linestyle='--', linewidth=1.5, label='Umbral Clínico (0.22)')
    ax1.set_ylabel('Eye Aspect Ratio (EAR)', fontsize=11, fontweight='bold')
    ax1.legend(loc='upper right')
    
    # --- PANEL INFERIOR: APM REAL ---
    ax2.plot(df_sys['timestamp'], df_sys['keystrokes'], color='#2980b9', linewidth=1.5, label='Teclas por Minuto')
    ax2.axhline(y=20, color='#e74c3c', linestyle='--', linewidth=1.5, label='Umbral Inactividad (20)')
    ax2.set_ylabel('Inactividad (APM)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Hora Local', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper right')

    # 4. Formatear el eje de tiempo para que se vea legible (HH:MM:SS)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.xticks(rotation=45)

    plt.suptitle('Telemetría Real: Fusión Sensorial Multimodal de ATOM', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # 5. Exportar a PNG de alta resolución
    plt.savefig('figura_atom_real.png', dpi=300, bbox_inches='tight')
    print("✅ Gráfica real generada con éxito: figura_atom_real.png")

except Exception as e:
    print(f"❌ Error al leer la base de datos: {e}")
