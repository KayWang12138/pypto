#!/usr/bin/env bash
# End-to-end: build op_kernel_lib.so from a pypto-exported model and run
# the op_probe binary against it. Positional arg 1 is the path to the
# model file (.onnx or .air) written by ../export_demo.py — same contract
# as the export script, so the path produced there gets copy-pasted here.
#
# Three steps:
#   1. load_and_compile.py extracts the op's cpp sources from the model
#      and builds op_kernel_lib.cpython-*.so against the local Ascend libs.
#   2. cmake configures + builds the op_probe binary in this dir. The
#      probe's libpython ABI is pinned to match the .so's cpython-XYZ tag
#      (otherwise the Python import machinery refuses to load the .so
#      with a generic "ModuleNotFoundError").
#   3. Run op_probe against the freshly built .so.
#
# Env overrides (all have per-example defaults):
#   PYTHON         python used to drive load_and_compile.py (default: python3
#                  from PATH — must have pypto installed; override if not)
#   OP_TYPE        custom op type to extract
#   KERNEL_BUILD   where load_and_compile.py writes the build tree
#   PROBE_BUILD    where this probe is configured + built
set -euo pipefail

usage() {
    echo "usage: $0 <model_path>" >&2
    echo "       model_path is the .onnx / .air file produced by ../export_demo.py" >&2
    exit 2
}
[[ $# -eq 1 ]] || usage
MODEL="$1"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../../../.." && pwd)"

PYTHON="${PYTHON:-python3}"
OP_TYPE="${OP_TYPE:-AddPyptoCustomOp}"
KERNEL_BUILD="${KERNEL_BUILD:-/tmp/pypto_session/build_op_kernel_lib_combined_add}"
PROBE_BUILD="${PROBE_BUILD:-/tmp/pypto_session/op_probe_combined_add_cpp}"

if [[ ! -f "${MODEL}" ]]; then
    echo "[FAIL] model not found: ${MODEL}" >&2
    exit 1
fi

echo "[STEP 1/3] load_and_compile.py -> op_kernel_lib.so (op=${OP_TYPE})"
"${PYTHON}" "${REPO_ROOT}/examples/export/load_and_compile.py" \
    "${MODEL}" \
    --op-type "${OP_TYPE}" \
    --out-dir "${KERNEL_BUILD}" \
    --clean

SO_PATH="$(ls "${KERNEL_BUILD}/build/"op_kernel_lib.*.so 2>/dev/null | head -n1 || true)"
if [[ -z "${SO_PATH}" ]]; then
    echo "[FAIL] no op_kernel_lib.*.so produced under ${KERNEL_BUILD}/build" >&2
    exit 1
fi
echo "[INFO] built .so: ${SO_PATH}"

# Parse the cpython ABI tag out of the .so filename
# ("op_kernel_lib.cpython-312-...so" -> python3.12) and pin the probe's
# libpython to the matching interpreter via -DPython_EXECUTABLE.
SO_TAG="$(basename "${SO_PATH}" | sed -n 's/^op_kernel_lib\.cpython-\([0-9]\+\)-.*\.so$/\1/p')"
PROBE_PY_FLAG=()
if [[ -n "${SO_TAG}" ]]; then
    MAJ="${SO_TAG:0:1}"
    MIN="${SO_TAG:1}"
    PY_CAND="$(command -v "python${MAJ}.${MIN}" || true)"
    if [[ -z "${PY_CAND}" && -x "/usr/bin/python${MAJ}.${MIN}" ]]; then
        PY_CAND="/usr/bin/python${MAJ}.${MIN}"
    fi
    if [[ -n "${PY_CAND}" ]]; then
        echo "[INFO] pinning probe to python${MAJ}.${MIN} (${PY_CAND}) to match .so ABI tag cpython-${SO_TAG}"
        PROBE_PY_FLAG=(-DPython_EXECUTABLE="${PY_CAND}")
    else
        echo "[WARN] .so needs python${MAJ}.${MIN} but it isn't on PATH; letting cmake autodetect" >&2
    fi
fi

echo "[STEP 2/3] cmake configure + build probe"
rm -rf "${PROBE_BUILD}"
cmake -S "${HERE}" -B "${PROBE_BUILD}" "${PROBE_PY_FLAG[@]}"
cmake --build "${PROBE_BUILD}" -j

PROBE_BIN="${PROBE_BUILD}/op_probe"
if [[ ! -x "${PROBE_BIN}" ]]; then
    echo "[FAIL] probe binary missing at ${PROBE_BIN}" >&2
    exit 1
fi

echo "[STEP 3/3] run probe against ${SO_PATH}"
echo "-----"
"${PROBE_BIN}" "${SO_PATH}"
