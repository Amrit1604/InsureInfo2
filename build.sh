#!/bin/bash
# Render.com build script
echo "🚀 Building LLM Claims Processing API for Render..."
echo "� Setting up Python environment..."

# Upgrade build tools first
pip install --upgrade pip setuptools wheel

echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

echo "🧪 Testing import compatibility..."
python -c "import fastapi, uvicorn, google.generativeai; print('✅ Core imports successful')"

echo "✅ Build complete!"
