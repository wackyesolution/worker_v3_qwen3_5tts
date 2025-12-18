#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python3}
VENV_PATH=${VENV_PATH:-.venv-azzurra}
CUDA_INDEX_URL=${CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu124}
TORCH_VERSION=${TORCH_VERSION:-2.4.0}
TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION:-2.4.0}
SKIP_VENV=${SKIP_VENV:-0}
SKIP_FFMPEG=${SKIP_FFMPEG:-0}

banner() {
  printf '\n[%s]\n' "$1"
}

install_ffmpeg() {
  if [[ "$SKIP_FFMPEG" == "1" ]]; then
    banner "Skipping FFmpeg installation (SKIP_FFMPEG=1)"
    return
  fi

  local apt_cmd=""
  if command -v apt >/dev/null 2>&1; then
    apt_cmd="apt"
  elif command -v apt-get >/dev/null 2>&1; then
    apt_cmd="apt-get"
  fi

  if [[ -z "$apt_cmd" ]]; then
    banner "FFmpeg install skipped (apt not available)"
    if command -v ffmpeg >/dev/null 2>&1; then
      ffmpeg -version | head -n 1 || true
    else
      echo "Install FFmpeg manually for your platform, then rerun this script." >&2
    fi
    return
  fi

  if [[ $EUID -ne 0 ]]; then
    echo "FFmpeg installation requires root privileges. Re-run as root or set SKIP_FFMPEG=1." >&2
    exit 1
  fi

  banner "Installing FFmpeg via $apt_cmd"
  $apt_cmd update
  $apt_cmd upgrade -y
  $apt_cmd install -y ffmpeg
  ffmpeg -version | head -n 1 || true
}

check_python() {
  banner "Checking Python interpreter ($PYTHON_BIN)"
  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python interpreter '$PYTHON_BIN' not found." >&2
    exit 1
  fi
  "$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit("Chatterblez requires Python 3.11 or newer.")
PY
}

setup_venv() {
  if [[ "$SKIP_VENV" == "1" ]]; then
    banner "Skipping virtualenv creation (SKIP_VENV=1)"
    return
  fi

  banner "Setting up virtualenv at $VENV_PATH"
  if [[ ! -d "$VENV_PATH" ]]; then
    "$PYTHON_BIN" -m venv "$VENV_PATH"
  fi
  # shellcheck disable=SC1090
  source "$VENV_PATH/bin/activate"
  PYTHON_BIN=python3
}

run_pip() {
  "$PYTHON_BIN" -m pip "$@"
}

install_base_tools() {
  banner "Upgrading pip/setuptools/wheel"
  run_pip install --upgrade pip setuptools wheel
}

install_torch_stack() {
  banner "Installing torch/torchaudio from $CUDA_INDEX_URL"
  local torch_spec="torch"
  local torchaudio_spec="torchaudio"
  if [[ -n "${TORCH_VERSION}" ]]; then
    torch_spec="torch==${TORCH_VERSION}"
  fi
  if [[ -n "${TORCHAUDIO_VERSION}" ]]; then
    torchaudio_spec="torchaudio==${TORCHAUDIO_VERSION}"
  fi
  run_pip install "$torch_spec" "$torchaudio_spec" --index-url "$CUDA_INDEX_URL"
}

install_project_requirements() {
  banner "Installing backend requirements for Azzurra"
  run_pip install -r requirements_azzurra.txt
}

install_ffmpeg
check_python
setup_venv
install_base_tools
install_torch_stack
install_project_requirements

banner "Azzurra environment ready! Activate with: source ${VENV_PATH}/bin/activate"
