#!/bin/bash

# Setup script for Shipboard Fire Response AI System
# This script sets up the development environment

set -e

echo "🚢 Setting up Shipboard Fire Response AI System..."

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8"

if ! python3 -c "import sys; assert sys.version_info >= (3, 8)" 2>/dev/null; then
    echo "❌ Python 3.8+ is required. Found: $python_version"
    exit 1
fi
echo "✅ Python version: $python_version"

# Create virtual environment
echo "🐍 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Install development dependencies
echo "🛠️ Installing development dependencies..."
pip install -e .[dev]

# Set up pre-commit hooks
echo "🔗 Setting up pre-commit hooks..."
pre-commit install

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p logs
mkdir -p models
mkdir -p data/raw
mkdir -p data/processed
mkdir -p uploads
mkdir -p static
mkdir -p templates

# Set up environment file
echo "⚙️ Setting up environment configuration..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✅ Environment file created (.env)"
    echo "📝 Please edit .env with your configuration"
else
    echo "✅ Environment file already exists"
fi

# Check for Docker
echo "🐳 Checking Docker installation..."
if command -v docker &> /dev/null; then
    echo "✅ Docker is installed"
    docker_version=$(docker --version)
    echo "   Version: $docker_version"
else
    echo "⚠️ Docker is not installed (optional for development)"
fi

# Check for MySQL
echo "🗄️ Checking MySQL installation..."
if command -v mysql &> /dev/null; then
    echo "✅ MySQL is available"
    mysql_version=$(mysql --version)
    echo "   Version: $mysql_version"
else
    echo "⚠️ MySQL not found. Install MySQL or use Docker for database"
fi

# Run initial tests
echo "🧪 Running initial tests..."
pytest tests/unit/ -v --tb=short || echo "⚠️ Some tests failed. This is normal for initial setup."

# Check code style
echo "🎨 Checking code style..."
black --check src/ tests/ || echo "⚠️ Code formatting issues found. Run 'black src/ tests/' to fix."

# Final setup message
echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your configuration"
echo "2. Set up your database (see docs/INSTALLATION.md)"
echo "3. Run 'source venv/bin/activate' to activate the environment"
echo "4. Run 'python -m shipboard_ai.api.server' to start the development server"
echo ""
echo "For more information, see:"
echo "- README.md for project overview"
echo "- docs/INSTALLATION.md for detailed setup"
echo "- docs/API.md for API documentation"
echo "- CONTRIBUTING.md for development guidelines"
echo ""
echo "Happy coding! 🚢🔥🤖"
