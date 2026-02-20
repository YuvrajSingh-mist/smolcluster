
#!/bin/bash
set -e

echo "=== Jetson PyTorch GPU Install Script ==="

# Check if passwordless sudo is configured
if ! sudo -n true 2>/dev/null; then
    echo "❌ ERROR: This script requires passwordless sudo access."
    echo ""
    echo "To configure passwordless sudo on your Jetson:"
    echo "  1. Run: sudo visudo"
    echo "  2. Add this line at the end (replace 'username' with your actual username):"
    echo "     username ALL=(ALL) NOPASSWD:ALL"
    echo "  3. Save and exit (Ctrl+X, then Y, then Enter)"
    echo ""
    echo "Alternatively, run these commands manually with sudo:"
    echo "  sudo apt update && sudo apt upgrade -y"
    echo "  sudo apt install -y python3-pip python3-venv libopenblas-base libopenmpi-dev python3.10 python3.10-venv python3.10-dev"
    echo ""
    echo "Then run this script again."
    exit 1
fi

echo "✅ Passwordless sudo configured"
echo "Updating system packages..."
sudo apt update -y
sudo apt upgrade -y

echo "Installing prerequisites..."
sudo apt install -y python3-pip python3-venv libopenblas-base libopenmpi-dev python3.10 python3.10-venv python3.10-dev

echo ""
echo "--- Removing any existing user torch packages ---"
pip3 uninstall -y torch torchvision torchaudio || true
pip3 uninstall -y torch torchvision torchaudio || true

echo ""
echo "=== Setting up project with uv ==="

# Get the project directory (two levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Project directory: $PROJECT_DIR"
cd "$PROJECT_DIR"

# Ensure uv is in PATH
export PATH=/opt/homebrew/bin:/usr/local/bin:$HOME/.cargo/bin:$HOME/.local/bin:$PATH

# Set CUDA environment variables for Jetson
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

echo "CUDA environment:"
echo "  CUDA_HOME: $CUDA_HOME"
echo "  LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install uv first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "✅ uv found: $(which uv)"

echo ""
echo "Navigating to project root: $PROJECT_DIR"
cd "$PROJECT_DIR"

# Pin Python version to 3.10
echo "📌 Pinning Python version to 3.10..."
uv python pin 3.10

# Check if .venv exists in project root
if [[ ! -d "$PROJECT_DIR/.venv" ]]; then
    echo "📦 Creating virtual environment with uv (Python 3.10) in project root..."
    uv venv --python 3.10
else
    echo "✅ Virtual environment already exists at $PROJECT_DIR/.venv"
    echo "🔍 Checking Python version in venv..."
    VENV_PYTHON_VERSION=$($PROJECT_DIR/.venv/bin/python3 --version 2>&1 | grep -oP '\d+\.\d+')
    if [[ "$VENV_PYTHON_VERSION" != "3.10" ]]; then
        echo "⚠️  Current venv is Python $VENV_PYTHON_VERSION, but Python 3.10 is required"
        echo "🔄 Removing old venv and creating new one with Python 3.10..."
        rm -rf "$PROJECT_DIR/.venv"
        uv venv --python 3.10
        echo "✅ New venv created with Python 3.10"
    fi
fi

echo ""
echo "--- Upgrading pip, setuptools, wheel in uv environment"
uv pip install --upgrade pip setuptools wheel

echo ""
echo "=== Running uv sync to install project dependencies ==="
uv sync

echo ""
echo "=== Removing default PyTorch before installing Jetson-specific version ==="
uv pip uninstall torch torchvision || true

echo ""
echo "=== Installing PyTorch + TorchVision from Jetson Wheel Index ==="
echo "This will override the PyTorch in pyproject.toml with Jetson-specific builds..."
echo "Python version: $($PROJECT_DIR/.venv/bin/python3 --version)"
echo "Python path: $($PROJECT_DIR/.venv/bin/python3 -c 'import sys; print(sys.executable)')"

# Check CUDA availability
echo ""
echo "Checking for CUDA libraries..."
if ldconfig -p | grep -q libcuda.so; then
    echo "✅ CUDA libraries found"
    ldconfig -p | grep libcuda.so | head -3
else
    echo "⚠️  WARNING: CUDA libraries not found in ldconfig"
fi

echo ""
echo "Using venv's pip directly for Jetson-specific installation..."

# Use the venv's pip directly with full path to ensure we're installing in the venv
uv pip install torch==2.8.0 torchvision==0.23.0 \
    --index-url https://pypi.jetson-ai-lab.io/jp6/cu126

echo ""
echo "=== Downgrading NumPy for compatibility with Jetson PyTorch ==="
echo "Jetson PyTorch 2.8.0 was compiled with NumPy 1.x..."
uv pip install "numpy<2"

echo ""
echo "=== Installation done! Verifying PyTorch CUDA ==="
echo "Checking installed wheel info..."
uv pip show torch | grep -E "(Name|Version|Location|Summary)"
echo ""

$PROJECT_DIR/.venv/bin/python3 - << 'EOF'
import torch
import os

print("=" * 60)
print("PyTorch Configuration:")
print("=" * 60)
print(f"Torch version: {torch.__version__}")
print(f"Torch file location: {torch.__file__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version (compiled): {torch.version.cuda if hasattr(torch.version, 'cuda') else 'N/A'}")

if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
else:
    print("⚠️  WARNING: CUDA not available!")
    print("Checking CUDA environment variables:")
    print(f"  CUDA_HOME: {os.environ.get('CUDA_HOME', 'not set')}")
    print(f"  LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'not set')}")
print("=" * 60)
EOF

echo ""
echo "== Done! To activate the environment, run:"
echo "   source $PROJECT_DIR/.venv/bin/activate"
echo ""
echo "Or use 'uv run' to run commands in the environment:"
echo "   cd $PROJECT_DIR && uv run python your_script.py"
echo ""
echo "Virtual environment location: $PROJECT_DIR/.venv"
echo ""
