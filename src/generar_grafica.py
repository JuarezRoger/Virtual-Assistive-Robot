import matplotlib.pyplot as plt
import numpy as np

# Configuración académica de la gráfica
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# 1. Generar eje de tiempo (60 segundos)
tiempo = np.arange(0, 60, 1)

# 2. Simular datos del Ojo (EAR)
# Comienza normal (~0.30), cae a fatiga visual en el segundo 35
ear = 0.32 + 0.03 * np.sin(tiempo / 3)
ear[35:45] = np.random.uniform(0.18, 0.21, 10) # Caída por debajo de 0.22
ear[45:] = 0.30 + 0.02 * np.random.randn(15)   # Recuperación tras la alerta de ATOM

# 3. Simular datos de Teclado (APM)
# Comienza activo (~50 APM), cae a casi 0 en el segundo 35
apm = 50 + 15 * np.cos(tiempo / 4)
apm[35:45] = np.random.uniform(0, 15, 10) # Caída por debajo de 20
apm[45:] = 45 + 5 * np.random.randn(15)   # Recuperación

# --- DIBUJAR GRÁFICA SUPERIOR (EAR) ---
ax1.plot(tiempo, ear, color='#2c3e50', linewidth=2, label='EAR Promedio (5s)')
ax1.axhline(y=0.22, color='#e74c3c', linestyle='--', linewidth=1.5, label='Umbral Clínico (0.22)')
ax1.set_ylabel('Eye Aspect Ratio (EAR)', fontsize=11, fontweight='bold')
ax1.set_ylim(0.15, 0.40)
ax1.legend(loc='upper right')

# --- DIBUJAR GRÁFICA INFERIOR (APM) ---
ax2.plot(tiempo, apm, color='#2980b9', linewidth=2, label='Teclas por Minuto (APM)')
ax2.axhline(y=20, color='#e74c3c', linestyle='--', linewidth=1.5, label='Umbral Inactividad (20)')
ax2.set_ylabel('Inactividad Motora (APM)', fontsize=11, fontweight='bold')
ax2.set_xlabel('Tiempo (Segundos)', fontsize=11, fontweight='bold')
ax2.set_ylim(0, 80)
ax2.legend(loc='upper right')

# --- RESALTAR LA ZONA DE INTERVENCIÓN (Compuerta AND) ---
# Sombreado rojo donde AMBAS condiciones se cumplen
ax1.axvspan(35, 45, color='#e74c3c', alpha=0.15)
ax2.axvspan(35, 45, color='#e74c3c', alpha=0.15)
ax1.text(40, 0.35, 'Intervención de ATOM\n(Fusión Sensorial)', 
         horizontalalignment='center', color='#c0392b', fontweight='bold',
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

# Ajustes finales y guardado en alta resolución
plt.suptitle('Fusión Sensorial Multimodal: Detección de Fatiga en Ventanas de 5 Segundos', fontsize=14, fontweight='bold')
plt.tight_layout()

# Guardar la imagen en calidad de publicación (300 dpi)
plt.savefig('figura_atom.png', dpi=300, bbox_inches='tight')
print("✅ Gráfica generada con éxito: figura_atom.png")