@echo off
REM Helper script to run setup from anywhere
REM This script maps the network drive and runs setup

echo Mapping network drive...
net use Z: "\\LTAds\DataStor\malezieux_meryl\Postdoc\Analysis\Python_scripts\HR_GUI_Project" /persistent:no

if errorlevel 1 (
    echo Failed to map network drive. Please check your network connection.
    pause
    exit /b 1
)

echo Changing to project directory...
Z:
cd "Z:\"

echo Running setup...
call setup_env.bat

REM Disconnect the mapped drive when done
net use Z: /delete

