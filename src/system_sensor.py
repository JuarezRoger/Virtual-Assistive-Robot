import ctypes
import time
import threading
from ctypes import wintypes
import psutil
import pygetwindow as gw
from pynput import keyboard, mouse
from datetime import datetime
import sqlite3
import os

# Ejecutable -> nombre legible. Lo no listado se guarda como el .exe sin extension.
NOMBRES_APP = {
    "chrome.exe": "Google Chrome", "msedge.exe": "Microsoft Edge", "firefox.exe": "Firefox",
    "code.exe": "Visual Studio Code", "windowsterminal.exe": "Terminal",
    "powershell.exe": "PowerShell", "pwsh.exe": "PowerShell", "cmd.exe": "Terminal",
    "explorer.exe": "Explorador de archivos", "python.exe": "Python (ATOM)",
    "pythonw.exe": "Python (ATOM)", "claude.exe": "Claude", "zoom.exe": "Zoom",
    "ms-teams.exe": "Teams", "teams.exe": "Teams", "blender.exe": "Blender",
    "winword.exe": "Word", "excel.exe": "Excel", "powerpnt.exe": "PowerPoint",
    "notepad.exe": "Bloc de notas", "spotify.exe": "Spotify", "discord.exe": "Discord",
    "whatsapp.exe": "WhatsApp", "acrobat.exe": "Acrobat", "acrord32.exe": "Acrobat",
}

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
        # Columna `app`: nombre del programa (lo unico que se resume o se envia a la IA)
        if "app" not in {r[1] for r in cursor.execute("PRAGMA table_info(system_telemetry)")}:
            cursor.execute("ALTER TABLE system_telemetry ADD COLUMN app TEXT")
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

    def _get_active_app(self):
        """Nombre del programa en primer plano, a partir de su proceso."""
        try:
            hwnd = ctypes.windll.user32.GetForegroundWindow()
            pid = wintypes.DWORD()
            ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            exe = psutil.Process(pid.value).name()
            return NOMBRES_APP.get(exe.lower(), exe.rsplit(".", 1)[0])
        except Exception:
            return None

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
                (timestamp, active_window, app, keystrokes, mouse_clicks, cpu_usage, ram_usage)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp, self.current_window, self._get_active_app(), self.keystrokes,
                  self.mouse_clicks, cpu_usage, ram_usage))
            
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
