@echo off
REM Git setup script for Heart Rate Detection GUI Project
REM This script helps set up git and push to GitHub

echo ========================================
echo GitHub Setup for HR Detection GUI
echo ========================================
echo.

REM Check if git is installed
git --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Git is not installed!
    echo Please install Git from: https://git-scm.com/download/win
    pause
    exit /b 1
)

echo Git is installed.
echo.

REM Initialize git if not already done
if not exist .git (
    echo Initializing git repository...
    git init
    echo Git repository initialized.
    echo.
) else (
    echo Git repository already initialized.
    echo.
)

REM Show current status
echo Current git status:
git status
echo.

echo ========================================
echo Next steps:
echo ========================================
echo 1. Review the files to be committed above
echo 2. Add files: git add .
echo 3. Commit: git commit -m "Initial commit"
echo 4. Add remote: git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
echo 5. Push: git push -u origin main
echo.
echo See GITHUB_SETUP.md for detailed instructions.
echo.

pause

