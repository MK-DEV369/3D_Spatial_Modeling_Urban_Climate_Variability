echo Services started. Close windows to stop. Use:
echo   docker compose -f \"%DC_FILE%\" down

@echo off
setlocal enabledelayedexpansion

REM === Urban Climate Platform: Start All Services ===
REM - Starts Docker services: db (PostGIS), redis, celery worker, celery beat (ephemeral)
REM - Starts backend (Django) in WSL with python3
REM - Starts frontend (Vite dev server)

set "ROOT=%~dp0"
set "DC_FILE=%ROOT%docker-compose.yml"

echo [0/4] Launching Docker Desktop...
start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe" >nul 2>&1

REM Wait for Docker daemon to be ready (max 60s)
echo [1/4] Waiting for Docker daemon to be ready...
set DOCKER_READY=0
for /l %%i in (1,1,30) do (
    docker info >nul 2>&1 && set DOCKER_READY=1 && goto :docker_ready
    timeout /t 2 >nul
)
:docker_ready
if !DOCKER_READY! neq 1 (
    echo WARNING: Docker daemon did not start in time. Continuing anyway...
)


REM Start Docker Compose services (db, redis, celery, celery-beat)
echo [2/4] Starting Docker Compose services...
docker compose up -d db redis celery celery-beat
if %errorlevel% neq 0 echo WARNING: Failed to start some Docker services. Continuing...

REM Launch backend in WSL (new terminal)
echo [3/4] Launching Backend (WSL)...
start "Backend (WSL)" wsl bash -c "cd backend && source venv/Scripts/activate && python3 manage.py runserver"

REM Launch frontend (new terminal)
echo [4/4] Launching Frontend...
start "Frontend" cmd /k "cd /d "%ROOT%frontend" && npm run dev"

echo ---
echo All services launched (attempted).
echo - Frontend: http://localhost:5173
echo - Backend API: http://localhost:8000/api/
echo - Docker services: db (5432), redis (6379), celery worker
echo ---
echo To stop all: docker compose -f "%DC_FILE%" down
endlocal
