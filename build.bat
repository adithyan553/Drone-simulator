@echo off
echo === Drone Flight Simulator - EXE build ===
python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found. Install Python 3.10+ from https://python.org and tick "Add Python to PATH".
    pause
    exit /b 1
)

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo Dependency install failed.
    pause
    exit /b 1
)

python -m PyInstaller --noconfirm --clean --onefile --windowed ^
    --name DroneFlightSimulator ^
    --collect-all ursina ^
    --collect-all panda3d ^
    --collect-all panda3d_gltf ^
    --collect-all panda3d_simplepbr ^
    --hidden-import PIL ^
    main.py
if errorlevel 1 (
    echo Build failed.
    pause
    exit /b 1
)

echo.
echo Done!  Your executable is at: dist\DroneFlightSimulator.exe
pause
