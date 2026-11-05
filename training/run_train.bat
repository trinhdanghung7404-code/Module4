@echo off
chcp 65001 > nul
echo ====================================================================
echo  HUẤN LUYỆN TẦNG NƠ-RON PHÂN LOẠI KHUYẾT TẬT GỐM SỨ (CD-MLP)
echo  Mô hình: DINOv2 (Frozen) + CeramicDefectHead (384 -^> 128 -^> 32 -^> 4)
echo ====================================================================
echo.

set PYTHON_CMD=python
if exist "..\..\Module4-main\.venv\Scripts\python.exe" (
    set PYTHON_CMD=..\..\Module4-main\.venv\Scripts\python.exe
)
if exist "..\.venv\Scripts\python.exe" (
    set PYTHON_CMD=..\.venv\Scripts\python.exe
)

echo Đang chạy bằng Python: %PYTHON_CMD%
echo.

cd /d "%~dp0"
"%PYTHON_CMD%" train.py

echo.
echo ====================================================================
echo  HOÀN TẤT! Trọng số đã lưu tại training/defect_head.pt và models/
echo  Biểu đồ báo cáo đã lưu tại training/figures/
echo ====================================================================
pause
