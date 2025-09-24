#!/usr/bin/env bash
set -euo pipefail

# Colors
RED="\033[0;31m"; GREEN="\033[0;32m"; YELLOW="\033[1;33m"; BLUE="\033[0;34m"; NC="\033[0m"

info() { echo -e "${BLUE}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
ok() { echo -e "${GREEN}[OK]${NC} $1"; }
err() { echo -e "${RED}[ERROR]${NC} $1"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_YML="$REPO_ROOT/environment.yml"
ENV_NAME="germinal"

# Args
PYROSETTA_PATH=""
PYROSETTA_MODE="path"  # one of: path|conda|installer
usage() {
  cat <<USAGE
Usage: bash scripts/install_env.sh [--pyrosetta <path>] [--help]

Options:
  --pyrosetta <path>   Path to PyRosetta wheel/tarball OR directory containing it
  -h, --help           Show this help message
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pyrosetta)
      arg="${2:-}"
      if [[ -z "$arg" ]]; then err "--pyrosetta requires a value (conda|installer|<path>)"; exit 2; fi
      case "$arg" in
        conda)
          PYROSETTA_MODE="conda"; PYROSETTA_PATH=""; shift 2;;
        installer)
          PYROSETTA_MODE="installer"; PYROSETTA_PATH=""; shift 2;;
        *)
          PYROSETTA_MODE="path"; PYROSETTA_PATH="$arg"; shift 2;;
      esac
      ;;
    -h|--help)
      usage; exit 0;;
    *)
      warn "Unknown argument: $1"; shift;;
  esac
done

# Progress bar helper using tqdm if available
progress() {
  local total_steps=$1
  local current_step=$2
  local message=$3
  if command -v python >/dev/null 2>&1; then
    python - <<PY
try:
    from tqdm import tqdm
    bar = tqdm(total=$total_steps, position=0, leave=True)
    bar.update($current_step)
    bar.set_description_str("$message")
    bar.close()
except Exception:
    print("$message ($current_step/$total_steps)")
PY
  else
    echo "$message ($current_step/$total_steps)"
  fi
}

step=0; total=18

progress $total $((++step)) "Checking conda availability"
if ! command -v conda >/dev/null 2>&1; then
  err "Conda not found. Please install Miniconda/Anaconda and re-run."
  exit 1
fi
ok "Conda found: $(conda --version)"

progress $total $((++step)) "Detecting NVIDIA GPU and CUDA toolkit"
GPU_INFO="none"; CUDA_VERSION=""; NVSMI_CUDA=""
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -n1 || true)
  NVSMI_CUDA=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9][0-9]*\.[0-9]*\).*/\1/p' | head -n1 || true)
  ok "Detected GPU: $GPU_INFO"
else
  warn "nvidia-smi not found; proceeding without GPU detection."
fi

if command -v nvcc >/dev/null 2>&1; then
  CUDA_VERSION=$(nvcc --version | awk '/release/ {print $6}' | sed 's/,//')
  ok "Detected nvcc CUDA: $CUDA_VERSION"
else
  warn "nvcc not found; will rely on conda CUDA packages."
fi

progress $total $((++step)) "Creating or updating conda environment: $ENV_NAME"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  warn "Environment $ENV_NAME exists. Updating from environment.yml..."
  conda env update -n "$ENV_NAME" -f "$ENV_YML" | cat
else
  conda env create -n "$ENV_NAME" -f "$ENV_YML" | cat
fi
ok "Environment resolved"

progress $total $((++step)) "Activating environment"
# shellcheck disable=SC1091
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
set -u
ok "Activated $ENV_NAME"

progress $total $((++step)) "Removing existing JAX packages (if any)"
pip uninstall -y jax jaxlib jax_plugins 2>/dev/null | cat || true

progress $total $((++step)) "Installing JAX CUDA wheels (phase 1)"
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html | cat || warn "JAX CUDA wheels install warning"

progress $total $((++step)) "Installing CUDA nvcc"
conda install -y -c nvidia cuda-nvcc | cat || warn "cuda-nvcc install skipped"

progress $total $((++step)) "Installing cuDNN"
conda install -y -c nvidia cudnn | cat || warn "cudnn install skipped"

progress $total $((++step)) "Pinning JAX and dependent packages"
pip install "jax==0.5.3" | cat || warn "JAX CPU pin warning"
pip install "dm-haiku==0.0.13" | cat || warn "dm-haiku pin warning"
pip install hydra-core | cat || warn "hydra-core install warning"
pip install omegaconf | cat || warn "omegaconf install warning"

progress $total $((++step)) "Installing JAX CUDA12 pip build"
pip install "jax[cuda12_pip]==0.5.3" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html | cat || warn "JAX CUDA12 pip install warning"

progress $total $((++step)) "Configuring CUDA library path for this env"
ACT_DIR="$CONDA_PREFIX/etc/conda/activate.d"
DEACT_DIR="$CONDA_PREFIX/etc/conda/deactivate.d"
mkdir -p "$ACT_DIR" "$DEACT_DIR"
cat > "$ACT_DIR/zz_cuda_paths.sh" <<'EOS'
# Ensure CUDA libraries are discoverable when this env is active
export _OLD_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
EOS
cat > "$DEACT_DIR/zz_cuda_paths.sh" <<'EOS'
# Restore original LD_LIBRARY_PATH on deactivate
if [ -n "$_OLD_LD_LIBRARY_PATH" ]; then
  export LD_LIBRARY_PATH="$_OLD_LD_LIBRARY_PATH"
  unset _OLD_LD_LIBRARY_PATH
else
  unset LD_LIBRARY_PATH
fi
EOS
ok "CUDA paths activation hooks installed"

progress $total $((++step)) "Installing PyTorch with matching CUDA (optional)"
if command -v nvidia-smi >/dev/null 2>&1; then
  TORCH_CUDA_VER="12.1"
  CUDA_MAJOR=""
  if [[ -n "$CUDA_VERSION" ]]; then
    CUDA_MAJOR="${CUDA_VERSION%%.*}"
  elif [[ -n "$NVSMI_CUDA" ]]; then
    CUDA_MAJOR="${NVSMI_CUDA%%.*}"
  fi
  if [[ "$CUDA_MAJOR" == "11" ]]; then
    TORCH_CUDA_VER="11.8"
  fi
  conda install -y pytorch pytorch-cuda=${TORCH_CUDA_VER} -c pytorch -c nvidia | cat || warn "PyTorch CUDA install skipped"
else
  conda install -y pytorch cpuonly -c pytorch | cat || warn "PyTorch CPU install skipped"
fi
ok "PyTorch installation attempted"

progress $total $((++step)) "Installing Chai-1 (chai_lab)"
pip install -U chai-lab | cat || {
  warn "chai-lab pip installation failed; attempting GitHub install"
  pip install "git+https://github.com/chai-lab/Chai-1.git#egg=chai_lab" | cat || warn "Chai-1 installation failed"
}
ok "Chai-1 installation attempted"

progress $total $((++step)) "Editable installs for local packages"
pip install -e colabdesign
pip install -e .
ok "Editable installs completed"

progress $total $((++step)) "Adding repository colabdesign to PYTHONPATH"
ACT_DIR="$CONDA_PREFIX/etc/conda/activate.d"
DEACT_DIR="$CONDA_PREFIX/etc/conda/deactivate.d"
mkdir -p "$ACT_DIR" "$DEACT_DIR"
cat > "$ACT_DIR/zz_pythonpath_colabdesign.sh" <<EOS
# Added by Germinal installer: include repository colabdesign on PYTHONPATH
export _OLD_PYTHONPATH="\$PYTHONPATH"
export PYTHONPATH="$REPO_ROOT/colabdesign:\$PYTHONPATH"
EOS
cat > "$DEACT_DIR/zz_pythonpath_colabdesign.sh" <<'EOS'
# Restore previous PYTHONPATH on deactivate
if [ -n "$_OLD_PYTHONPATH" ]; then
  export PYTHONPATH="$_OLD_PYTHONPATH"
  unset _OLD_PYTHONPATH
else
  unset PYTHONPATH
fi
EOS
ok "colabdesign added to PYTHONPATH via activation hook"

progress $total $((++step)) "Installing PyRosetta (optional, may prompt license)"
python -m pip install pyrosetta-installer | cat || true
PYTMP="$(mktemp)"
cat > "$PYTMP" <<'PY'
try:
    import pyrosetta_installer
    pyrosetta_installer.install_pyrosetta()
    print("[OK] PyRosetta installed via pyrosetta-installer")
except Exception as e:
    print("[WARN] Skipping PyRosetta install:", e)
PY
python "$PYTMP" || true
rm -f "$PYTMP"

progress $total $((++step)) "Running validation script"
python "$REPO_ROOT/scripts/validate_install.py" || {
  err "Validation failed. See messages above."
  exit 2
}
ok "Validation succeeded"

echo
ok "Installation complete. Activate with: conda activate $ENV_NAME"
