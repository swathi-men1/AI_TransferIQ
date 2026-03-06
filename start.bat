@echo off
echo ============================================
echo    AI TransferIQ - Full Stack Launcher
echo ============================================

echo.
echo [1/2] Starting FastAPI Backend (port 8000)...
start "AI-TransferIQ Backend" cmd /k "uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000"

timeout /t 2 >nul

echo [2/2] Starting React Frontend (port 5173)...
start "AI-TransferIQ Frontend" cmd /k "cd /d frontend && npm run dev"

echo.
echo Both servers are starting!
echo   Frontend -> http://localhost:5173
echo   API Docs -> http://localhost:8000/docs
echo.
pause
