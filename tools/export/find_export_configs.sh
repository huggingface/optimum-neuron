#!/bin/bash
# Wrapper script to run find_export_configs.py in the optimum-neuron virtual environment
#
# This script:
# 1. Locates the workspace root
# 2. Creates/activates a .venv virtual environment
# 3. Installs optimum-neuron with neuronx extras if needed
# 4. Sets memory limits using ulimit
# 5. Runs the Python configuration testing script
#
# Usage:
#   ./find_export_configs.sh <model_id> [additional_args...]
#
# Environment Variables:
#   MAX_MEMORY_GB: Maximum memory in GB to allocate (default: 90% of available)
#
# Examples:
#   ./find_export_configs.sh Qwen/Qwen3-Embedding-8B
#   MAX_MEMORY_GB=100 ./find_export_configs.sh meta-llama/Llama-3.1-8B --dry-run

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
echo "Activating virtual environment in ${VENV_PATH}..."
source "${VENV_PATH}/bin/activate"

# Install optimum-neuron with neuronx extras if not already installed
echo "Checking dependencies..."
python -c "import optimum.neuron; import neuronx_distributed" 2>/dev/null || {
    echo "Installing optimum-neuron with neuronx extras..."
    pip install -e "${WORKSPACE_ROOT}[neuronx]" > /dev/null 2>&1
}

# Change to workspace root for consistent relative paths
cd "${WORKSPACE_ROOT}"

# Set memory limits using ulimit
echo ""
echo "Configuring memory limits..."
if [ -n "${MAX_MEMORY_GB}" ]; then
    # User specified memory limit
    MEMORY_LIMIT_GB="${MAX_MEMORY_GB}"
    echo "Using user-specified memory limit: ${MEMORY_LIMIT_GB}GB"
else
    # Calculate 90% of available memory
    TOTAL_MEMORY_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    TOTAL_MEMORY_GB=$((TOTAL_MEMORY_KB / 1024 / 1024))
    MEMORY_LIMIT_GB=$((TOTAL_MEMORY_GB * 90 / 100))
    echo "Detected total memory: ${TOTAL_MEMORY_GB}GB"
    echo "Using 90% of available memory: ${MEMORY_LIMIT_GB}GB"
fi

# Convert GB to KB for ulimit
MEMORY_LIMIT_KB=$((MEMORY_LIMIT_GB * 1024 * 1024))

# Set virtual memory limit
if ulimit -v "${MEMORY_LIMIT_KB}" 2>/dev/null; then
    echo "✓ Set virtual memory limit: ${MEMORY_LIMIT_GB}GB"
else
    echo "⚠ Warning: Could not set memory limit (ulimit -v not supported or insufficient permissions)"
    echo "  The process will run without memory constraints"
fi

# Set data segment limit as additional safeguard
if ulimit -d "${MEMORY_LIMIT_KB}" 2>/dev/null; then
    echo "✓ Set data segment limit: ${MEMORY_LIMIT_GB}GB"
else
    echo "⚠ Warning: Could not set data segment limit (may not be supported on this system)"
fi

# Report current limits
echo ""
echo "Active resource limits:"
echo "  Virtual memory: $(ulimit -v) KB"
echo "  Data segment:   $(ulimit -d) KB"

# Run the Python script with all arguments
echo ""
echo "Starting configuration test..."
echo "=========================================="
python "${PYTHON_SCRIPT}" "$@"
