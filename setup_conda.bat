@echo off
REM Setup script for Heart Rate Detection GUI Project using Conda
REM Creates a conda environment and installs dependencies

echo Creating conda environment 'hr_detection_gui'...
conda create -n hr_detection_gui python=3.9 -y

echo Activating environment...
call conda activate hr_detection_gui

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing requirements...
pip install -r requirements.txt

echo.
echo Setup complete!
echo.
echo To activate the environment in the future, run:
echo   conda activate hr_detection_gui
echo.
echo To run the application:
echo   python main.py
echo.
echo To deactivate when done:
echo   conda deactivate
echo.

pause

