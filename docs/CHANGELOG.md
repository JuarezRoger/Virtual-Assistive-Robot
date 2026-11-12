# ATOM - Registro de Cambios

## v2.0.0 — 2026-04-07: ML Predictivo + Ergonomía HPE

### Nuevos módulos

**fatigue_model.py — Modelo Predictivo de Fatiga (Random Forest)**
- Reemplaza el sistema de umbrales fijos (EAR < 0.22) con un clasificador Random Forest que aprende los patrones de fatiga personales del usuario
- Se entrena con datos fusionados: EAR, variabilidad de EAR, tasa de parpadeos, APM, clics, y deltas respecto al baseline matutino
- Las etiquetas provienen del NASA-TLX: registros con score compuesto > 60 se clasifican como "fatigado"
- Incluye validación cruzada, análisis de importancia de features, y persistencia del modelo a disco (pickle)
- Fallback automático a heurísticas si no hay suficientes datos para entrenar (mínimo 10 muestras con ambas clases)
- Método `get_adaptive_threshold()`: calcula un umbral EAR personalizado usando el percentil 15 del historial del usuario

**Dependencia nueva:** `scikit-learn>=1.6`

### Módulos modificados

**visual_sensor.py — Migración de dlib a MediaPipe + HPE + Blinks**
- Motor de visión migrado de dlib (68 landmarks) a MediaPipe FaceMesh (468 landmarks). Elimina la dependencia de `shape_predictor_68_face_landmarks.dat`
- Head Pose Estimation (HPE) implementado con `cv2.solvePnP` usando 6 puntos faciales clave contra un modelo 3D genérico. Calcula pitch (inclinación frontal), yaw (rotación lateral) y roll (inclinación lateral) en grados
- Detección de parpadeos: cuenta blinks por intervalo usando EAR < 0.21 sostenido durante >= 2 frames consecutivos
- Migración automática de esquema: agrega columnas `blinks`, `pitch`, `yaw`, `roll` a `visual_telemetry` si no existen
- Guardado agrupado cada 10 segundos (antes se guardaba cada frame, generando ~27k registros por sesión)
- HUD en tiempo real: overlay semitransparente con EAR, blinks, pitch, yaw, roll y códigos de color
- Callbacks de alerta de postura: dispara cuando pitch > 25° o yaw > 30°
- Clase renombrada internamente pero mantiene el nombre `VisualSensorPro` para compatibilidad con `lanzador_atom.py`

**agent_trigger.py — Motor de Intervención Inteligente v2.0**
- Integración con `FatiguePredictor`: usa probabilidad ML (umbral >= 65%) en lugar de comparaciones fijas
- Nuevo tipo de alerta: postura cervical. Si pitch o yaw exceden umbrales durante la ventana de observación, genera intervención específica con recomendaciones ergonómicas vía GPT
- API key movida de hardcoded a variable de entorno `OPENAI_API_KEY`
- Cooldown diferenciado: 60s para fatiga, 120s para postura
- Ventana de análisis ampliada de ~5 segundos a 1 minuto completo
- Log mejorado: muestra modo (ML/Heurístico/Fallback) y probabilidad en cada ciclo
- Fallback gracioso si el modelo ML no está disponible

**data_fusion.py — Fusión Multimodal v2.0**
- Compatible con esquema viejo (solo avg_ear) y nuevo (blinks, pitch, yaw, roll)
- Nuevas métricas derivadas: `ear_std` (variabilidad), `blink_rate`, `posture_score` (magnitud vectorial de pitch+yaw), `cognitive_load_index` (índice compuesto 0-1)
- Visualización multi-panel: hasta 4 paneles (EAR, APM, Blinks, Postura) con tema oscuro profesional
- Exportación automática a CSV fusionado

### Esquema de base de datos

Tabla `visual_telemetry` — columnas agregadas:
- `blinks INTEGER DEFAULT 0` — parpadeos detectados en el intervalo
- `pitch REAL DEFAULT 0.0` — inclinación frontal de la cabeza (grados)
- `yaw REAL DEFAULT 0.0` — rotación lateral de la cabeza (grados)
- `roll REAL DEFAULT 0.0` — inclinación lateral de la cabeza (grados)

La migración es automática: `visual_sensor.py` ejecuta `ALTER TABLE` al inicializarse si detecta columnas faltantes.

### Seguridad

- API key de OpenAI removida del código fuente. Ahora se lee desde `os.environ.get("OPENAI_API_KEY")`
- Se recomienda configurar la variable de entorno antes de ejecutar

---

## v1.0.0 — 2026-03-11: Arquitectura Inicial

- Sensor de sistema: keystrokes, mouse clicks, CPU, RAM, ventana activa
- Sensor visual: EAR con dlib 68 landmarks
- Dashboard: CustomTkinter con métricas EAR/APM y gráfica pie
- Trigger: Alertas GPT-4o-mini con umbrales fijos (EAR < 0.22, APM < 20)
- Lanzador: Orquestador de microservicios con reinicio automático
- NASA-TLX: Formulario integrado en el dashboard
