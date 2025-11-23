@echo off
REM Setup script for Heart Rate Detection GUI Project
REM Maps network drive and creates virtual environment

echo Mapping network drive...
net use Z: "\\LTAds\DataStor\malezieux_meryl\Postdoc\Analysis\Python_scripts\HR_GUI_Project" /persistent:no

echo Changing to project directory...
Z:
cd "Z:\"

echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing requirements...
pip install -r requirements.txt

echo.
echo Setup complete!
echo.
echo To activate the environment in the future:
echo   1. Map the drive: net use Z: "\\LTAds\DataStor\malezieux_meryl\Postdoc\Analysis\Python_scripts\HR_GUI_Project"
echo   2. Change to: Z:
echo   3. Activate: venv\Scripts\activate.bat
echo.
echo To run the application:
echo   python main.py
echo.

REM Disconnect the mapped drive
net use Z: /delete

pause

