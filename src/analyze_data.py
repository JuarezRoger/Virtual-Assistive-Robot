import sqlite3
import pandas as pd

def analyze_telemetry(db_path="../data/atom_telemetry.db"):
    # Conectar a la base de datos
    conn = sqlite3.connect(db_path)
    
    # Leer los datos en un DataFrame de Pandas
    query = "SELECT * FROM system_telemetry"
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        print("La base de datos está vacía. Deja correr el sensor un rato más.")
        return

    # Convertir la columna timestamp a tipo datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Mostrar un resumen de las aplicaciones más usadas
    print("\n--- Top 5 Ventanas Más Activas ---")
    top_windows = df['active_window'].value_counts().head(5)
    print(top_windows)
    
    # Mostrar el promedio de actividad por registro (10 seg)
    print("\n--- Promedio de Actividad ---")
    print(f"Teclas por intervalo: {df['keystrokes'].mean():.2f}")
    print(f"Clics por intervalo: {df['mouse_clicks'].mean():.2f}")
    print(f"Uso de CPU promedio: {df['cpu_usage'].mean():.2f}%")

if __name__ == "__main__":
    analyze_telemetry()