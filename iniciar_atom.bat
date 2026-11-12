@echo off
title ATOM - Sistema de Monitoreo Cognitivo

echo.
echo  =========================================
echo   ATOM - Sistema de Monitoreo Cognitivo
echo  =========================================
echo.

REM Ir a la raiz del proyecto
cd /d "%~dp0"

REM Verificar que el venv existe
if not exist "venv\Scripts\python.exe" (
    echo  [ERROR] No se encontro el entorno virtual en: venv\
    echo  Ejecuta primero: python -m venv venv
    pause
    exit /b 1
)

REM Verificar si mediapipe esta instalado
venv\Scripts\python.exe -c "import mediapipe" 2>nul
if errorlevel 1 (
    echo  [INFO] Instalando paquetes compatibles...
    echo  [INFO] numpy 1.26.4 + protobuf 4.25.3 + mediapipe 0.10.33
    venv\Scripts\pip install numpy==1.26.4 protobuf==4.25.3 mediapipe==0.10.33
    echo  [OK] Paquetes listos.
    echo.
)

REM Forzar UTF-8 en todos los subprocesos Python
set PYTHONIOENCODING=utf-8

REM Lanzar ATOM desde src\ usando el Python del venv
echo  Iniciando sensores...
echo.
cd src
"..\venv\Scripts\python.exe" lanzador_atom.py
