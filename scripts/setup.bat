@echo off
REM Setup script for Shipboard Fire Response AI System (Windows)
REM This script sets up the development environment on Windows

echo 🚢 Setting up Shipboard Fire Response AI System...

REM Check Python version
echo 📋 Checking Python version...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set python_version=%%i
echo ✅ Python version: %python_version%

REM Create virtual environment
echo 🐍 Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo ✅ Virtual environment created
) else (
    echo ✅ Virtual environment already exists
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo 📦 Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo 📥 Installing dependencies...
pip install -r requirements.txt

REM Install development dependencies
echo 🛠️ Installing development dependencies...
pip install -e .[dev]

REM Set up pre-commit hooks
echo 🔗 Setting up pre-commit hooks...
pre-commit install

REM Create necessary directories
echo 📁 Creating necessary directories...
if not exist "logs" mkdir logs
if not exist "models" mkdir models
if not exist "data\raw" mkdir data\raw
if not exist "data\processed" mkdir data\processed
if not exist "uploads" mkdir uploads
if not exist "static" mkdir static
if not exist "templates" mkdir templates

REM Set up environment file
echo ⚙️ Setting up environment configuration...
if not exist ".env" (
    copy .env.example .env
    echo ✅ Environment file created (.env)
    echo 📝 Please edit .env with your configuration
) else (
    echo ✅ Environment file already exists
)

REM Check for Docker
echo 🐳 Checking Docker installation...
docker --version >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Docker is installed
    for /f "tokens=*" %%i in ('docker --version') do echo    Version: %%i
) else (
    echo ⚠️ Docker is not installed (optional for development)
)

REM Run initial tests
echo 🧪 Running initial tests...
pytest tests/unit/ -v --tb=short
if %errorlevel% neq 0 (
    echo ⚠️ Some tests failed. This is normal for initial setup.
)

REM Check code style
echo 🎨 Checking code style...
black --check src/ tests/
if %errorlevel% neq 0 (
    echo ⚠️ Code formatting issues found. Run 'black src/ tests/' to fix.
)

REM Final setup message
echo.
echo 🎉 Setup complete!
echo.
echo Next steps:
echo 1. Edit .env file with your configuration
echo 2. Set up your database (see docs/INSTALLATION.md)
echo 3. Run 'venv\Scripts\activate.bat' to activate the environment
echo 4. Run 'python -m shipboard_ai.api.server' to start the development server
echo.
echo For more information, see:
echo - README.md for project overview
echo - docs/INSTALLATION.md for detailed setup
echo - docs/API.md for API documentation
echo - CONTRIBUTING.md for development guidelines
echo.
echo Happy coding! 🚢🔥🤖

pause
