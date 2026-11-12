import subprocess
import time
import sys
from datetime import datetime

try:
    import winsound
    def hacer_pitido():
        winsound.Beep(1000, 500) 
except ImportError:
    def hacer_pitido():
        print('\a', end='', flush=True)

# ¡AQUÍ ESTABA EL ERROR! Cambiamos la "y" por la "i"
sensores = [
    "system_sensor.py",   # <-- Nombre exacto de tu archivo en VS Code
    "visual_sensor.py",
    "live_plot.py",
    "agent_trigger.py"
]

procesos = {} 
nombre_log = f"atom_logs_{datetime.now().strftime('%Y%m%d')}.txt"
archivo_log = open(nombre_log, "a", encoding="utf-8")

def iniciar_sensor(sensor):
    # Sin rutas extra, ya que todo está en la misma carpeta 'src'
    if sensor in ["agent_trigger.py", "live_plot.py"]:
        return subprocess.Popen([sys.executable, sensor]) 
    else:
        return subprocess.Popen(
            [sys.executable, sensor], 
            stdout=archivo_log,
            stderr=subprocess.STDOUT
        )

try:
    print("🤖 Iniciando la arquitectura ATOM (Modo Rápido)...")
    archivo_log.write(f"\n--- Inicio de sesión ATOM: {datetime.now().strftime('%H:%M:%S')} ---\n")
    
    for sensor in sensores:
        procesos[sensor] = iniciar_sensor(sensor)
        print(f"✅ [{sensor}] en línea.")
    
    print("\n⚡ ATOM operando. Monitoreando caídas...")
    print("Presiona Ctrl+C en esta terminal para apagar todo de forma segura.\n")
    print("-" * 50)
    
    while True:
        for sensor, proceso in procesos.items():
            if proceso.poll() is not None:
                print(f"\n⚠️ ¡Alerta! [{sensor}] se ha desconectado. Reiniciando en 5 segundos...")
                archivo_log.write(f"--- Reinicio automático de {sensor}: {datetime.now().strftime('%H:%M:%S')} ---\n")
                
                hacer_pitido()
                time.sleep(5) 
                
                procesos[sensor] = iniciar_sensor(sensor)
                print(f"🔄 [{sensor}] reiniciado exitosamente.\n" + "-" * 50)
        
        time.sleep(2)

except KeyboardInterrupt:
    print("\n🛑 Apagando la red de sensores ATOM...")
    archivo_log.write(f"--- Apagado de sesión ATOM: {datetime.now().strftime('%H:%M:%S')} ---\n")
    
    for sensor, proceso in procesos.items():
        if proceso.poll() is None: 
            proceso.terminate()
            proceso.wait()
    
    archivo_log.close()
    print("💤 Todos los microservicios terminados de forma segura.")
