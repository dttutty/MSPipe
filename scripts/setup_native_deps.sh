#!/usr/bin/env bash
set -euo pipefail

MINIFORGE_PREFIX="${MINIFORGE_PREFIX:-$HOME/miniforge3}"
NATIVE_ENV_NAME="${NATIVE_ENV_NAME:-mspipe-native}"
CUDA_VERSION="${CUDA_VERSION:-11.8}"
ABSEIL_VERSION="${ABSEIL_VERSION:-20220623.0}"
RMM_SPEC="${RMM_SPEC:-librmm=23.06.*}"
THRUST_SPEC="${THRUST_SPEC:-thrust=1.16.*}"
CUB_SPEC="${CUB_SPEC:-cub=1.16.*}"

CONDA_BIN="${MINIFORGE_PREFIX}/bin/conda"

log() {
  printf '[setup_native_deps] %s\n' "$*"
}

die() {
  printf '[setup_native_deps] %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<EOF
Usage:
  bash scripts/setup_native_deps.sh install
  bash scripts/setup_native_deps.sh env

Commands:
  install   Install Miniforge if needed, then create/update the native
            dependency environment with fmt, spdlog, abseil-cpp, cmake,
            ninja, and librmm for CUDA ${CUDA_VERSION}.
  env       Print the environment exports needed by setup.py / CMake.

Environment overrides:
  MINIFORGE_PREFIX   Default: ${MINIFORGE_PREFIX}
  NATIVE_ENV_NAME    Default: ${NATIVE_ENV_NAME}
  CUDA_VERSION       Default: ${CUDA_VERSION}
  ABSEIL_VERSION     Default: ${ABSEIL_VERSION}
  RMM_SPEC           Default: ${RMM_SPEC}
  THRUST_SPEC        Default: ${THRUST_SPEC}
  CUB_SPEC           Default: ${CUB_SPEC}
EOF
}

need_command() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

native_prefix() {
  printf '%s/envs/%s' "${MINIFORGE_PREFIX}" "${NATIVE_ENV_NAME}"
}

download_file() {
  local url="$1"
  local out="$2"

  if command -v curl >/dev/null 2>&1; then
    curl -fsSL -o "${out}" "${url}"
    return
  fi
  if command -v wget >/dev/null 2>&1; then
    wget -qO "${out}" "${url}"
    return
  fi
  die "Need curl or wget to download Miniforge"
}

ensure_miniforge() {
  if [[ -x "${CONDA_BIN}" ]]; then
    log "Using existing Miniforge at ${MINIFORGE_PREFIX}"
    return
  fi

  local os arch installer url
  os="$(uname)"
  arch="$(uname -m)"
  [[ "${os}" == "Linux" ]] || die "This bootstrap script currently supports Linux only"

  installer="${TMPDIR:-/tmp}/Miniforge3-${os}-${arch}.sh"
  url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-${os}-${arch}.sh"

  log "Downloading Miniforge installer from ${url}"
  download_file "${url}" "${installer}"

  log "Installing Miniforge into ${MINIFORGE_PREFIX}"
  bash "${installer}" -b -p "${MINIFORGE_PREFIX}"
}

ensure_native_env() {
  local prefix
  prefix="$(native_prefix)"

  if [[ -d "${prefix}" ]]; then
    log "Updating native dependency environment ${NATIVE_ENV_NAME}"
    "${CONDA_BIN}" install -y -n "${NATIVE_ENV_NAME}" -c conda-forge \
      cmake ninja fmt spdlog "abseil-cpp=${ABSEIL_VERSION}"
    return
  fi

  log "Creating native dependency environment ${NATIVE_ENV_NAME}"
  "${CONDA_BIN}" create -y -n "${NATIVE_ENV_NAME}" -c conda-forge \
    cmake ninja fmt spdlog "abseil-cpp=${ABSEIL_VERSION}"
}

install_rmm() {
  log "Installing ${RMM_SPEC}, ${THRUST_SPEC}, and ${CUB_SPEC} for CUDA ${CUDA_VERSION}"

  if "${CONDA_BIN}" install -y -n "${NATIVE_ENV_NAME}" -c rapidsai -c conda-forge \
      "${RMM_SPEC}" "${THRUST_SPEC}" "${CUB_SPEC}" "cuda-version=${CUDA_VERSION}"; then
    return
  fi

  log "Falling back to rapidsai label cuda-${CUDA_VERSION}.0"
  if "${CONDA_BIN}" install -y -n "${NATIVE_ENV_NAME}" \
      -c "rapidsai/label/cuda-${CUDA_VERSION}.0" -c conda-forge \
      "${RMM_SPEC}" "${THRUST_SPEC}" "${CUB_SPEC}"; then
    return
  fi

  log "Falling back to rapidsai label cuda-${CUDA_VERSION}"
  if "${CONDA_BIN}" install -y -n "${NATIVE_ENV_NAME}" \
      -c "rapidsai/label/cuda-${CUDA_VERSION}" -c conda-forge \
      "${RMM_SPEC}" "${THRUST_SPEC}" "${CUB_SPEC}"; then
    return
  fi

  die "Failed to install ${RMM_SPEC}. Try a different CUDA_VERSION, RMM_SPEC, THRUST_SPEC, or CUB_SPEC."
}

print_env() {
  local prefix
  prefix="$(native_prefix)"

  cat <<EOF
export PATH="${prefix}/bin\${PATH:+:\$PATH}"
export RMM_DIR="${prefix}/lib/cmake/rmm"
export ABSL_DIR="${prefix}/lib/cmake/absl"
export SPDLOG_DIR="${prefix}/lib/cmake/spdlog"
export FMT_DIR="${prefix}/lib/cmake/fmt"
export CMAKE_PREFIX_PATH="${prefix}\${CMAKE_PREFIX_PATH:+:\$CMAKE_PREFIX_PATH}"
export LD_LIBRARY_PATH="${prefix}/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
EOF
}

install() {
  need_command bash
  need_command uname

  if ! command -v nvcc >/dev/null 2>&1; then
    log "nvcc not found on PATH. Install CUDA ${CUDA_VERSION} separately before building gnnflow."
  fi

  ensure_miniforge
  ensure_native_env
  install_rmm

  log "Native dependencies installed into $(native_prefix)"
  printf '\n'
  log 'Next step: eval "$(bash scripts/setup_native_deps.sh env)"'
}

main() {
  local cmd="${1:-install}"
  case "${cmd}" in
    install)
      install
      ;;
    env)
      print_env
      ;;
    -h|--help|help)
      usage
      ;;
    *)
      usage >&2
      exit 1
      ;;
  esac
}

main "$@"
