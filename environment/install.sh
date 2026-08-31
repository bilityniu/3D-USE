#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
NERFSTUDIO_ROOT="${NERFSTUDIO_ROOT:-${ROOT}/third_party/nerfstudio}"
PYTHON_BIN="${PYTHON:-python3}"
export MAX_JOBS="${MAX_JOBS:-2}"
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export PYTORCH_JIT=0

if [[ -z "${CC:-}" ]] && command -v gcc-10 >/dev/null 2>&1; then
  export CC="$(command -v gcc-10)"
fi
if [[ -z "${CXX:-}" ]] && command -v g++-10 >/dev/null 2>&1; then
  export CXX="$(command -v g++-10)"
fi
if [[ -z "${CUDAHOSTCXX:-}" ]] && command -v g++-10 >/dev/null 2>&1; then
  export CUDAHOSTCXX="$(command -v g++-10)"
fi

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || {
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
}
"${PYTHON_BIN}" - <<'PY'
import sys

if sys.version_info[:2] != (3, 11):
    raise SystemExit(
        f"3D-USE requires Python 3.11; "
        f"received {sys.version.split()[0]}"
    )
PY

for required in \
  "${NERFSTUDIO_ROOT}/nerfstudio/__init__.py" \
  "${NERFSTUDIO_ROOT}/nerfstudio/data/datamanagers/base_datamanager.py" \
  "${NERFSTUDIO_ROOT}/nerfstudio/data/dataparsers/base_dataparser.py" \
  "${NERFSTUDIO_ROOT}/nerfstudio/data/datasets/base_dataset.py" \
  "${ROOT}/threeduse/cuda/include/glm/glm.hpp"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing bundled dependency source: ${required}" >&2
    echo "Re-clone the complete 3D-USE repository." >&2
    exit 1
  fi
done

"${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
"${PYTHON_BIN}" -m pip install -r "${ROOT}/environment/nerfstudio-main-requirements.txt"

# The repository includes the modified Nerfstudio source used by 3D-USE.
# gsplat is an unmodified dependency and is installed from its official
# package; the project-owned underwater compositor lives in threeduse/cuda.
BUILD_NO_CUDA=1 "${PYTHON_BIN}" -m pip install "gsplat==1.5.3" --no-build-isolation
"${PYTHON_BIN}" -m pip install -e "${NERFSTUDIO_ROOT}" --no-deps --no-build-isolation
"${PYTHON_BIN}" -m pip install -e "${ROOT}" --no-build-isolation

"${PYTHON_BIN}" "${ROOT}/scripts/validate_install.py"
