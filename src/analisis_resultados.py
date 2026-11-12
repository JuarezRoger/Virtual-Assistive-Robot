import sqlite3
import pandas as pd
from datetime import timedelta

print("📊 INICIANDO ANÁLISIS ACADÉMICO DE ATOM...\n")

db_path = "../data/atom_telemetry.db"

try:
    conn = sqlite3.connect(db_path)
    # Cargar toda la base de datos
    df_sys = pd.read_sql_query("SELECT timestamp, keystrokes, active_window FROM system_telemetry", conn)
    df_vis = pd.read_sql_query("SELECT timestamp, avg_ear FROM visual_telemetry", conn)
    conn.close()

    df_sys['timestamp'] = pd.to_datetime(df_sys['timestamp'])
    df_vis['timestamp'] = pd.to_datetime(df_vis['timestamp'])

    # 1. Encontrar Intervenciones (EAR < 0.22 y APM < 20)
    # Simplificaremos cruzando los datos por el minuto exacto para encontrar coincidencias
    df_sys['minuto'] = df_sys['timestamp'].dt.floor('Min')
    df_vis['minuto'] = df_vis['timestamp'].dt.floor('Min')
    
    # Promediar por minuto
    sys_min = df_sys.groupby('minuto')['keystrokes'].mean().reset_index()
    vis_min = df_vis.groupby('minuto')['avg_ear'].mean().reset_index()
    
    df_merged = pd.merge(sys_min, vis_min, on='minuto')
    
    # Filtrar donde se cumplió la compuerta AND
    intervenciones = df_merged[(df_merged['avg_ear'] < 0.22) & (df_merged['keystrokes'] < 20)]
    total_alertas = len(intervenciones)

    print("--- 1. MÉTRICAS DE INTERVENCIÓN Y TOKENOMICS ---")
    print(f"Total de alertas detonadas: {total_alertas}")
    
    # Cálculo de viabilidad económica (Asumiendo ~60 tokens por alerta con gpt-4o-mini a $0.15 / 1M tokens)
    tokens_usados = total_alertas * 60
    costo_total = (tokens_usados / 1000000) * 0.15
    print(f"Tokens consumidos aprox.: {tokens_usados}")
    print(f"Costo operativo de la API: ${costo_total:.6f} USD")

    print("\n--- 2. IMPACTO FISIOLÓGICO (DELTA DE RECUPERACIÓN) ---")
    if total_alertas > 0:
        ear_previo_total = 0
        ear_post_total = 0
        
        for index, row in intervenciones.iterrows():
            minuto_alerta = row['minuto']
            
            # Buscar EAR 3 minutos antes (Fatiga progresiva)
            mask_antes = (df_vis['timestamp'] >= (minuto_alerta - timedelta(minutes=3))) & (df_vis['timestamp'] < minuto_alerta)
            ear_antes = df_vis.loc[mask_antes, 'avg_ear'].mean()
            
            # Buscar EAR 3 minutos después (Recuperación post-intervención)
            mask_despues = (df_vis['timestamp'] > minuto_alerta) & (df_vis['timestamp'] <= (minuto_alerta + timedelta(minutes=3)))
            ear_despues = df_vis.loc[mask_despues, 'avg_ear'].mean()
            
            ear_previo_total += ear_antes if pd.notna(ear_antes) else row['avg_ear']
            ear_post_total += ear_despues if pd.notna(ear_despues) else row['avg_ear']
            
        avg_ear_previo = ear_previo_total / total_alertas
        avg_ear_post = ear_post_total / total_alertas
        mejora_porcentual = ((avg_ear_post - avg_ear_previo) / avg_ear_previo) * 100
        
        print(f"EAR Promedio Pre-Intervención (Fatiga): {avg_ear_previo:.3f}")
        print(f"EAR Promedio Post-Intervención (Recuperación): {avg_ear_post:.3f}")
        print(f"Delta de Mejora Ocular: {mejora_porcentual:.1f}%")
    else:
        print("No se registraron alertas suficientes para calcular el delta.")

except Exception as e:
    print(f"Error procesando datos: {e}")