#!/bin/bash
# Setup script to install OpenMM with CUDA support for uv environment

set -e

CONDA_ENV_NAME="openmm-cuda"
CONDA_PREFIX="$HOME/miniconda3"
CONDA_ENV_PATH="$CONDA_PREFIX/envs/$CONDA_ENV_NAME"
UV_VENV=".venv"

echo "Setting up OpenMM with CUDA support..."

# Check if conda is available
if [ ! -f "$CONDA_PREFIX/bin/conda" ]; then
    echo "Error: Miniconda not found at $CONDA_PREFIX"
    echo "Please install Miniconda first:"
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh -b -p $CONDA_PREFIX"
    exit 1
fi

# Create conda environment if it doesn't exist
if [ ! -d "$CONDA_ENV_PATH" ]; then
    echo "Creating conda environment: $CONDA_ENV_NAME"
    $CONDA_PREFIX/bin/conda create -n $CONDA_ENV_NAME python=3.12 -c conda-forge -y
    
    echo "Installing OpenMM with CUDA support..."
    $CONDA_PREFIX/bin/conda run -n $CONDA_ENV_NAME conda install -c conda-forge openmm "cuda-version>=12.8,<13" -y
else
    echo "Conda environment $CONDA_ENV_NAME already exists"
fi

# Check if uv venv exists
if [ ! -d "$UV_VENV" ]; then
    echo "Creating uv virtual environment..."
    uv sync
fi

# Pin uv to use conda Python
echo "Configuring uv to use conda Python..."
uv python pin $CONDA_ENV_PATH/bin/python

# Copy conda OpenMM into uv venv
echo "Copying OpenMM with CUDA support into uv venv..."
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
SITE_PACKAGES="$UV_VENV/lib/python$PYTHON_VERSION/site-packages"

if [ -d "$SITE_PACKAGES" ]; then
    # Remove existing OpenMM if present
    rm -rf "$SITE_PACKAGES/openmm"* "$SITE_PACKAGES/simtk" "$SITE_PACKAGES/OpenMM.libs" 2>/dev/null || true
    
    # Copy conda OpenMM
    cp -r "$CONDA_ENV_PATH/lib/python$PYTHON_VERSION/site-packages/openmm"* "$SITE_PACKAGES/"
    cp -r "$CONDA_ENV_PATH/lib/python$PYTHON_VERSION/site-packages/simtk" "$SITE_PACKAGES/" 2>/dev/null || true
    
    echo "OpenMM with CUDA support installed successfully!"
else
    echo "Error: Could not find site-packages directory: $SITE_PACKAGES"
    exit 1
fi

# Verify installation
echo ""
echo "Verifying CUDA support..."
export LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib:$LD_LIBRARY_PATH"
uv run python -c "import openmm; platforms = [openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())]; print('Available platforms:', platforms)" || {
    echo "Warning: CUDA platform not detected. Make sure to set LD_LIBRARY_PATH:"
    echo "  export LD_LIBRARY_PATH=$CONDA_ENV_PATH/lib:\$LD_LIBRARY_PATH"
}

echo ""
echo "Setup complete! To use CUDA, run:"
echo "  export LD_LIBRARY_PATH=$CONDA_ENV_PATH/lib:\$LD_LIBRARY_PATH"
echo "  uv run python your_script.py"

