@echo off
chcp 65001 >nul
echo ========================================
echo  Adaptive RAG 生产模式启动
echo ========================================
echo.

REM 确保数据目录存在
if not exist "data" mkdir data
if not exist "data\documents" mkdir data\documents
if not exist "data\qdrant_storage" mkdir data\qdrant_storage

REM 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python
    pause
    exit /b 1
)

echo 正在启动服务...
echo 地址: http://localhost:8000
echo 文档: http://localhost:8000/docs
echo.

REM 生产模式：无重载、单 worker、INFO 日志级别
python -m uvicorn adaptive_rag.api.main:app --host 127.0.0.1 --port 8000 --no-access-log --log-level info

pause
