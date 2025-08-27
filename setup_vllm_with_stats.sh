#!/bin/bash

# Setup script for modified vLLM with stats logging
# This script installs the local vLLM in development mode

set -e  # Exit on error

echo "🚀 Setting up modified vLLM with stats logging..."

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_DIR="$SCRIPT_DIR"

echo "📁 Working in directory: $VLLM_DIR"

# Check if we're in the right directory
if [ ! -f "$VLLM_DIR/setup.py" ] || [ ! -f "$VLLM_DIR/pyproject.toml" ]; then
    echo "❌ Error: This doesn't appear to be a vLLM directory"
    echo "   Expected to find setup.py and pyproject.toml"
    exit 1
fi

# Check Python version
echo "🐍 Checking Python version..."
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "   Python version: $python_version"

if [ "$python_version" != "3.8" ] && [ "$python_version" != "3.9" ] && [ "$python_version" != "3.10" ] && [ "$python_version" != "3.11" ]; then
    echo "⚠️  Warning: vLLM typically requires Python 3.8-3.11, you have $python_version"
fi

# Check if CUDA is available
echo "🔧 Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "   CUDA detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -n 5
else
    echo "   ⚠️  CUDA not detected - vLLM will run in CPU mode (very slow)"
fi

# Uninstall existing vLLM if present
echo "🧹 Checking for existing vLLM installation..."
if python3 -c "import vllm" 2>/dev/null; then
    echo "   Found existing vLLM installation. Uninstalling..."
    pip uninstall -y vllm || true
else
    echo "   No existing vLLM installation found."
fi

# Install dependencies
echo "📦 Installing build dependencies..."
pip install -U pip setuptools wheel
pip install ninja  # For faster compilation
pip install packaging  # Required for setup

# Install in development mode
echo "🔨 Installing modified vLLM in development mode..."
cd "$VLLM_DIR"

# Set environment variables for better compilation
export MAX_JOBS=4  # Limit parallel jobs to avoid memory issues
export NVCC_APPEND_FLAGS="-t 4"  # Limit NVCC threads

# Install in editable mode
pip install -e . --verbose

echo "✅ Installation completed!"

# Verify installation
echo "🧪 Verifying installation..."
if python3 -c "import vllm; print(f'vLLM version: {vllm.__version__}'); print('✅ Import successful!')" 2>/dev/null; then
    echo "✅ vLLM installation verified successfully!"
else
    echo "❌ Installation verification failed!"
    exit 1
fi

# Test our custom wrapper
echo "🧪 Testing custom stats wrapper..."
if python3 -c "from vllm.offline_llm_with_stats import OfflineLLMWithStats; print('✅ Custom wrapper import successful!')" 2>/dev/null; then
    echo "✅ Custom wrapper verified successfully!"
else
    echo "❌ Custom wrapper verification failed!"
    exit 1
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📝 Next steps:"
echo "   1. Test the installation: python3 test_vllm_stats_logging.py"
echo "   2. Use in your experiments with: from vllm.offline_llm_with_stats import OfflineLLMWithStats"
echo ""
echo "💡 Usage example:"
echo "   from vllm.offline_llm_with_stats import OfflineLLMWithStats"
echo "   from vllm import SamplingParams"
echo ""
echo "   llm = OfflineLLMWithStats(model='your/model/path', log_stats=True)"
echo "   outputs = llm.generate(['Hello world'], SamplingParams())"
echo ""
