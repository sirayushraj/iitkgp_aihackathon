@echo off
REM Run the Streamlit Dashboard locally
cd /d "%~dp0"
echo ============================================
echo   JETENGINE AI - PREDICTIVE MAINTENANCE
echo   Dashboard Starting...
echo ============================================
echo.

REM Try to find Python
set PYTHON_PATH=
if exist "C:\Users\ayush\AppData\Local\Programs\Python\Python314\python.exe" (
    set PYTHON_PATH=C:\Users\ayush\AppData\Local\Programs\Python\Python314\python.exe
) else (
    where python >nul 2>&1
    if %ERRORLEVEL% == 0 (
        set PYTHON_PATH=python
    ) else (
        where py >nul 2>&1
        if %ERRORLEVEL% == 0 (
            set PYTHON_PATH=py
        ) else (
            echo ERROR: Python not found!
            echo Please install Python or update PYTHON_PATH in this file.
            pause
            exit /b 1
        )
    )
)

echo Using Python: %PYTHON_PATH%
echo.

REM Check if dependencies are installed
echo Checking dependencies...
%PYTHON_PATH% -m pip show streamlit >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Installing dependencies...
    %PYTHON_PATH% -m pip install -r requirements.txt -q
)

echo.
echo Starting Streamlit Dashboard...
echo Dashboard will open in your browser at http://localhost:8501
echo Press Ctrl+C to stop the server
echo.
%PYTHON_PATH% -m streamlit run app.py

pause
