@echo off
chcp 65001 >nul
title Soi Khuyết Tật Gốm Sứ (AI DINOv2)

set "PYTHON_EXE=C:\Users\Admin\Documents\Module4-main\.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    set "PYTHON_EXE=python"
)

echo ======================================================================
echo    KHOI DONG HE THONG SOI LOI GOM SU BANG AI DINOv2
echo ======================================================================

"%PYTHON_EXE%" "%~dp0inspect_image.py" %*

pause
