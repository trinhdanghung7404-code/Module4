@echo off
title AI Capture Server - Module 4 Return Inspection
chcp 65001 >nul
echo ======================================================================
echo    KHỞI ĐỘNG AI CAPTURE SERVER (MODULE 4 RETURN INSPECTION)
echo ======================================================================
cd /d "%~dp0"

set "PYTHON_EXE=C:\Users\Admin\Documents\Module4-main\.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    set "PYTHON_EXE=python"
)

"%PYTHON_EXE%" app.py
pause
