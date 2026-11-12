import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.dates import DateFormatter

# Configuración de la gráfica
plt.style.use('dark_background') # Estilo hacker/ingeniero
fig, ax = plt.subplots(figsize=(10, 5))
db_path = "../data/atom_telemetry.db"

def animate(i):
    try:
        conn = sqlite3.connect(db_path)
        # Leer los últimos 60 registros (aprox 1 minuto de datos)
        df = pd.read_sql_query("SELECT timestamp, avg_ear FROM visual_telemetry ORDER BY id DESC LIMIT 60", conn)
        conn.close()
        
        if not df.empty:
            df = df.iloc[::-1] # Invertir para que el más reciente esté a la derecha
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            ax.clear()
            ax.plot(df['timestamp'], df['avg_ear'], color='#00ff00', marker='o', markersize=3, linestyle='-', linewidth=2)
            
            # Línea de Umbral (EAR: 0.21) - Por debajo de esto ATOM dispara alerta
            ax.axhline(y=0.21, color='red', linestyle='--', linewidth=1, label='Umbral de Fijación/Fatiga')
            
            ax.set_title('Telemetría Cognitiva en Tiempo Real (Dlib 68 pts)', fontsize=14, color='white')
            ax.set_ylabel('Eye Aspect Ratio (EAR)', fontsize=12, color='white')
            ax.set_xlabel('Tiempo', fontsize=12, color='white')
            ax.set_ylim(0.10, 0.40) # Rango normal del ojo humano
            
            ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
            plt.xticks(rotation=45)
            plt.tight_layout()
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
    except Exception as e:
        print(f"Esperando datos en la DB... ({e})")

# Iniciar la animación refrescando cada 1000 ms (1 segundo)
ani = animation.FuncAnimation(fig, animate, interval=1000, cache_frame_data=False)
plt.show()
