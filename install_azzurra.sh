#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python3}
VENV_PATH=${VENV_PATH:-.venv-azzurra}
CUDA_INDEX_URL=${CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu124}
TORCH_VERSION=${TORCH_VERSION:-2.4.0}
TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION:-2.4.0}
SKIP_VENV=${SKIP_VENV:-0}
SKIP_FFMPEG=${SKIP_FFMPEG:-0}
SKIP_TORCH=${SKIP_TORCH:-0}
USE_SYSTEM_SITE_PACKAGES=${USE_SYSTEM_SITE_PACKAGES:-0}

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
    venv_args=()
    if [[ "$USE_SYSTEM_SITE_PACKAGES" == "1" ]]; then
      venv_args+=(--system-site-packages)
    fi
    "$PYTHON_BIN" -m venv "${venv_args[@]}" "$VENV_PATH"
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
  if [[ "$SKIP_TORCH" == "1" ]]; then
    banner "Skipping torch/torchaudio installation (SKIP_TORCH=1)"
    if [[ "$USE_SYSTEM_SITE_PACKAGES" != "1" ]]; then
      echo "Warning: torch install skipped without system-site-packages. Ensure torch is available." >&2
    fi
    return
  fi
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

install_csm_rs() {
  if [[ "${INSTALL_CSM_RS:-1}" != "1" ]]; then
    banner "Skipping csm.rs build (INSTALL_CSM_RS!=1)"
    return
  fi

  local build_dir=${CSM_BUILD_DIR:-csm.rs}
  local repo=${CSM_REPO:-https://github.com/cartesia-one/csm.rs}
  local binary_name=${CSM_BINARY_NAME:-main}
  local features=${CSM_FEATURES:-cuda}
  local cargo_bin=${CARGO_BIN:-cargo}

  banner "Ensuring pkg-config and OpenSSL headers are installed"
  if command -v apt >/dev/null 2>&1 || command -v apt-get >/dev/null 2>&1; then
    local apt_cmd="apt"
    command -v apt-get >/dev/null 2>&1 && apt_cmd="apt-get"
    if [[ $EUID -ne 0 ]]; then
      echo "pkg-config/libssl-dev install requires root. Run with sudo or set INSTALL_CSM_RS=0." >&2
      exit 1
    fi
    $apt_cmd update
    $apt_cmd install -y pkg-config libssl-dev
  else
    echo "pkg-config/libssl-dev install skipped (apt not available). Install them manually if missing." >&2
  fi

  # Install Rust if cargo is not found
  if ! command -v "$cargo_bin" >/dev/null 2>&1; then
    banner "Installing Rust (rustup)"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    export PATH="$HOME/.cargo/bin:$PATH"
    cargo_bin="$HOME/.cargo/bin/cargo"
  fi

  banner "Building csm.rs ($features)"
  if [[ ! -d "$build_dir/.git" ]]; then
    rm -rf "$build_dir"
    git clone "$repo" "$build_dir"
  else
    git -C "$build_dir" pull --ff-only
  fi
  pushd "$build_dir" >/dev/null
  "$cargo_bin" build --release --features "$features"
  popd >/dev/null

  local binary_path="$build_dir/target/release/$binary_name"
  if [[ ! -f "$binary_path" ]]; then
    echo "Compilazione csm.rs completata ma binario '$binary_path' mancante." >&2
    exit 1
  fi
  local resolved_binary
  resolved_binary=$(realpath "$binary_path")
  local env_file=${CSM_ENV_FILE:-.azzurra-env}
  cat >"$env_file" <<EOF
# Source this file to use the Cartesia engine with FastAPI/CLI
export CHATTERBLEZ_TTS_ENGINE=csm
export CHATTERBLEZ_CSM_BINARY=$resolved_binary
export CHATTERBLEZ_CSM_MODEL=${CHATTERBLEZ_CSM_MODEL:-cartesia/azzurra-voice}
# export CHATTERBLEZ_CSM_EXTRA_ARGS="--cpu"   # optional
EOF
  cat <<EOV

[csm.rs]
Binary path       : $resolved_binary
Env helper file   : $env_file  (run: source $env_file)
EOV
}

install_ffmpeg
check_python
setup_venv
install_base_tools
install_torch_stack
install_project_requirements
install_csm_rs

banner "Azzurra environment ready! Activate with: source ${VENV_PATH}/bin/activate"
