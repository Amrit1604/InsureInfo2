#!/bin/bash
# Render.com build script - Python 3.13 compatibility fix
echo "🚀 Building LLM Claims Processing API for Render..."
echo "🔧 Python version: $(python --version)"

# Force upgrade build tools with specific versions
echo "📦 Upgrading build tools..."
pip install --upgrade pip==25.2
pip install --upgrade setuptools==75.1.0 wheel==0.43.0

# Install build dependencies first
echo "🔨 Installing build dependencies..."
pip install --no-cache-dir cython numpy

# Install requirements with no build isolation
echo "📦 Installing application dependencies..."
pip install --no-cache-dir --no-build-isolation -r requirements.txt

echo "🧪 Testing imports..."
python -c "
try:
    import fastapi, uvicorn, google.generativeai, numpy, sentence_transformers
    print('✅ All core imports successful')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

echo "✅ Build complete!"ender.com build script
echo "🚀 Building LLM Claims Processing API for Render..."
echo "� Setting up Python environment..."

# Upgrade build tools first
pip install --upgrade pip setuptools wheel

echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

echo "🧪 Testing import compatibility..."
python -c "import fastapi, uvicorn, google.generativeai; print('✅ Core imports successful')"

echo "✅ Build complete!"
