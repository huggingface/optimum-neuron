#!/bin/bash
# Wrapper script to run find_export_configs.py in the optimum-neuron virtual environment
#
# This script:
# 1. Locates the workspace root
# 2. Creates/activates a .venv virtual environment
# 3. Installs optimum-neuron with neuronx extras if needed
# 4. Runs the Python configuration testing script
#
# Usage:
#   ./find_export_configs.sh <model_id> [additional_args...]
#
# Examples:
#   ./find_export_configs.sh Qwen/Qwen3-Embedding-8B
#   ./find_export_configs.sh meta-llama/Llama-3.1-8B --dry-run

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find workspace root (go up two levels from tools/export)
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Construct paths
VENV_PATH="${WORKSPACE_ROOT}/.venv"
PYTHON_SCRIPT="${SCRIPT_DIR}/find_export_configs.py"

echo "=========================================="
echo "Export Configuration Testing"
echo "=========================================="
echo "Workspace root: ${WORKSPACE_ROOT}"
echo "Virtual environment: ${VENV_PATH}"
echo "Python script: ${PYTHON_SCRIPT}"
echo ""

# Check if workspace root exists and contains optimum_neuron structure
if [ ! -f "${WORKSPACE_ROOT}/pyproject.toml" ]; then
    echo "Error: Could not locate optimum-neuron workspace (no pyproject.toml found)"
    echo "Expected workspace root: ${WORKSPACE_ROOT}"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "${VENV_PATH}" ]; then
    echo "Creating virtual environment at ${VENV_PATH}..."
    python3 -m venv "${VENV_PATH}"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "${VENV_PATH}/bin/activate"

# Install optimum-neuron with neuronx extras if not already installed
echo "Checking dependencies..."
python -c "import optimum_neuron; import neuronx_distributed" 2>/dev/null || {
    echo "Installing optimum-neuron with neuronx extras..."
    pip install -e "${WORKSPACE_ROOT}[neuronx]" > /dev/null 2>&1
}

# Change to workspace root for consistent relative paths
cd "${WORKSPACE_ROOT}"

# Run the Python script with all arguments
echo ""
echo "Starting configuration test..."
echo "=========================================="
python "${PYTHON_SCRIPT}" "$@"
