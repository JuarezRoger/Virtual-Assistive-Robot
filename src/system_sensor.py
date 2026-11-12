import time
import threading
import psutil
import pygetwindow as gw
from pynput import keyboard, mouse
from datetime import datetime
import sqlite3
import os

class SystemSensor:
    def __init__(self, log_interval=10, db_path="../data/atom_telemetry.db"):
        self.log_interval = log_interval
        self.is_running = False
        self.db_path = db_path
        
        # Contadores de métricas
        self.keystrokes = 0
        self.mouse_clicks = 0
        self.current_window = ""
        
        # Inicializar la base de datos
        self._setup_database()

    def _setup_database(self):
        # Asegurar que la carpeta 'data' exista
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Conectar y crear la tabla si no existe
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_telemetry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                active_window TEXT,
                keystrokes INTEGER,
                mouse_clicks INTEGER,
                cpu_usage REAL,
                ram_usage REAL
            )
        ''')
        conn.commit()
        conn.close()
        print(f"Base de datos lista en: {self.db_path}")

    def _on_press(self, key):
        self.keystrokes += 1

    def _on_click(self, x, y, button, pressed):
        if pressed:
            self.mouse_clicks += 1

    def _get_active_window(self):
        try:
            window = gw.getActiveWindow()
            return window.title if window else "Desconocida"
        except Exception:
            return "Error capturando ventana"

    def _log_metrics(self):
        # Es vital abrir la conexión dentro del hilo que hará las inserciones
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        while self.is_running:
            time.sleep(self.log_interval)
            
            self.current_window = self._get_active_window()
            cpu_usage = psutil.cpu_percent()
            ram_usage = psutil.virtual_memory().percent
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Insertar el registro en la base de datos
            cursor.execute('''
                INSERT INTO system_telemetry 
                (timestamp, active_window, keystrokes, mouse_clicks, cpu_usage, ram_usage)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (timestamp, self.current_window, self.keystrokes, self.mouse_clicks, cpu_usage, ram_usage))
            
            conn.commit() # Confirmar los cambios
            
            safe_win = self.current_window[:20].encode('cp1252', errors='replace').decode('cp1252')
            print(f"[{timestamp}] Log DB -> Ventana: '{safe_win}...' | Teclas: {self.keystrokes} | Clics: {self.mouse_clicks}", flush=True)
            
            # Reiniciar contadores para el siguiente bloque de tiempo
            self.keystrokes = 0
            self.mouse_clicks = 0
            
        conn.close()

    def start(self):
        print("Iniciando Sensor de Sistema ATOM (Modo SQLite)...")
        self.is_running = True
        
        self.kb_listener = keyboard.Listener(on_press=self._on_press)
        self.mouse_listener = mouse.Listener(on_click=self._on_click)
        
        self.kb_listener.start()
        self.mouse_listener.start()
        
        self.log_thread = threading.Thread(target=self._log_metrics)
        self.log_thread.daemon = True
        self.log_thread.start()

    def stop(self):
        print("\nDeteniendo Sensor de Sistema y guardando datos...")
        self.is_running = False
        self.kb_listener.stop()
        self.mouse_listener.stop()

# --- Bloque de ejecución principal ---
if __name__ == "__main__":
    # Importante: Como el script corre desde src/, apuntamos db_path a la carpeta data/
    sensor = SystemSensor(log_interval=10, db_path="../data/atom_telemetry.db")
    sensor.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        sensor.stop()
        print("Sensor apagado de forma segura.")
