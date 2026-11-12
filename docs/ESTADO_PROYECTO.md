# ATOM - Estado del Proyecto
**Ultima actualizacion:** 2026-04-07

---

## Que funciona ahora mismo

| Componente | Estado | Notas |
|---|---|---|
| `system_sensor.py` | Funcionando | Captura teclado, mouse, CPU, ventana activa |
| `visual_sensor.py` | Funcionando (MODO DEMO) | Genera datos simulados realistas; modo real requiere MediaPipe |
| `agent_trigger.py` | Funcionando | Motor de alertas activo en modo heuristico |
| `lanzador_atom.py` | Funcionando | Orquesta los 4 microservicios con reinicio automatico |
| `atom_dashboard.py` | Funcionando | GUI con metricas en vivo y NASA-TLX |
| `data_fusion.py` | Funcionando | Fusion multimodal con 67+ minutos de datos |
| `fatigue_model.py` | Parcial | Logica lista; necesita mas datos NASA-TLX para entrenar |
| `iniciar_atom.bat` | Funcionando | Lanzador con venv integrado |

---

## Que falla y por que

### visual_sensor.py - Modo real no disponible
**Error:** `ModuleNotFoundError: No module named 'mediapipe'`

**Causa:** MediaPipe 0.10.9 fue descontinuado. La version en PyPI mas baja disponible es 0.10.13. El requirements.txt fue actualizado a `mediapipe==0.10.21` pero el paquete no ha sido instalado en el venv todavia.

**Solucion pendiente:**
```
cd C:\ATOM_Project
venv\Scripts\pip install mediapipe==0.10.21
```

**Estado actual:** El sensor entra automaticamente en MODO DEMO y sigue alimentando la base de datos con datos simulados realistas. El resto del sistema (trigger, dashboard, fusion) funciona normalmente con esos datos.

---

### fatigue_model.py - Modelo ML sin entrenar
**Estado:** El codigo esta completo y funcional, pero el modelo Random Forest no puede entrenarse todavia.

**Causa:** Solo hay 2 registros NASA-TLX en la base de datos, ambos con valores similares (~50 en todas las escalas). El modelo necesita minimo 10 registros con variacion entre estado descansado y fatigado.

**Solucion:** Registrar evaluaciones NASA-TLX periodicamente durante las sesiones de trabajo, especialmente:
- Al inicio de la jornada (cuando estas fresco)
- Despues de 2-3 horas de trabajo intenso (cuando sientes fatiga)
- Al final del dia

Una vez con suficientes datos, ejecutar:
```
cd C:\ATOM_Project\src
..\venv\Scripts\python.exe fatigue_model.py
```

**Estado actual:** El sistema usa umbrales heuristicos como fallback. Sigue funcionando, solo no es personalizado aun.

---

### API Key de OpenAI no configurada
**Estado:** El trigger detecta "API GPT: No configurada" y no genera mensajes de voz/texto contextuales.

**Causa:** La API key fue removida del codigo por seguridad. Ahora se lee desde una variable de entorno.

**Solucion:** Configurar la variable de entorno antes de lanzar ATOM:
```
set OPENAI_API_KEY=tu_clave_aqui
```
O agregarla permanentemente en Variables de Entorno del sistema Windows.

---

## Que queda por implementar (roadmap)

### Prioridad Alta
- [ ] **Instalar MediaPipe** para activar el modo real con camara
- [ ] **Recolectar mas datos NASA-TLX** para entrenar el modelo RF
- [ ] **Configurar API Key** en variables de entorno

### Prioridad Media
- [ ] **Dashboard v2**: Agregar tarjetas de postura (pitch/yaw) y blinks al `atom_dashboard.py`
- [ ] **live_plot.py v2**: Extender la grafica en tiempo real para mostrar blinks y postura ademas del EAR
- [ ] **Curva de aprendizaje ML**: Una vez con el modelo entrenado, mostrar como el umbral se personaliza con el tiempo

### Prioridad Baja
- [ ] **Exportar reporte PDF** de sesion automaticamente al cerrar ATOM
- [ ] **Calibracion inicial**: Asistente de primer uso que establece el baseline EAR personal en los primeros 5 minutos
- [ ] **Modo multiusuario**: Perfiles separados para distintos usuarios en la misma maquina

---

## Como ejecutar para una demostracion HOY

1. Abrir terminal en `C:\ATOM_Project`
2. Ejecutar `iniciar_atom.bat`
3. El sistema arranca con:
   - Sensor de sistema capturando actividad real de teclado/mouse
   - Sensor visual en MODO DEMO generando EAR/blinks/postura simulados
   - Motor de alertas monitoreando y disparando si el EAR simulado baja de 0.22
   - Dashboard mostrando metricas en vivo
4. Para ver la grafica de fusion: `venv\Scripts\python.exe src\data_fusion.py`
5. Para registrar una evaluacion NASA-TLX: boton en el dashboard

---

## Arquitectura actual

```
lanzador_atom.py
    |
    +-- system_sensor.py   -> BD: system_telemetry
    |                            (keystrokes, mouse, CPU, ventana)
    |
    +-- visual_sensor.py   -> BD: visual_telemetry
    |   [DEMO o REAL]           (avg_ear, blinks, pitch, yaw, roll)
    |
    +-- agent_trigger.py   -> Lee ambas tablas cada 5s
    |   [ML o heuristico]       Llama GPT si detecta fatiga/mala postura
    |
    +-- live_plot.py        -> Grafica EAR en tiempo real

atom_dashboard.py           -> GUI independiente (lanzar por separado)
fatigue_model.py            -> Entrena/carga modelo RF (lanzar manualmente)
data_fusion.py              -> Reporte de sesion (lanzar manualmente)
```
