# 故障排除（高频）

## 快速索引

**🔧 环境加载/变量问题** — 最常见的根因
- 🔥 [npu-smi 运行失败：libc_sec.so](#npu-smi-运行失败libc_secso-not-found)
- 🔥 [torch_npu 导入失败：共享库找不到](#torch_npu-导入失败共享库找不到-libhcclso--libatbso--libascend_halso)
- 🔥 [PYTHONPATH 导致 pypto import 异常](#pythonpath-导致-pypto-import-异常)

**📦 安装/编译问题**
- 🔥 [pypto 导入失败：DT_FP8E8M0 缺失](#pypto-导入失败dt_fp8e8m0-缺失)
- ⚡ [PTO ISA 编译/头文件错误](#pto-isa-编译头文件错误)
- ⚡ [pto-isa 版本不匹配](#pto-isa-版本不匹配缺少-ptotrowexpandadd--ptotrowexpandmax)

**🔄 版本/依赖冲突**
- ⚡ [undefined symbol / ABI 不匹配](#undefined-symbol--abi-不匹配)
- ⚡ [pip 依赖冲突](#pip-依赖冲突resolutionimpossible)
- 💤 [conda 中 torch 存在但 import 失败](#conda-中-torch-存在但-import-失败)

**⬇️ 下载问题**
- 💤 [下载/安装包损坏](#下载安装包损坏unexpected-archive-size)

排查前先完成通用检查：
1. **运行诊断脚本**：`python3 "$SKILL_DIR/scripts/diagnose_env.py" --pretty`
2. **确认 Python 解释器**：`which python3`
3. **确认 CANN 已加载**：`echo $ASCEND_HOME_PATH`
4. **若用 conda**：确认已激活正确 env

## npu-smi 运行失败：libc_sec.so not found

现象（示例）：
- `npu-smi: error while loading shared libraries: libc_sec.so: cannot open shared object file: No such file or directory`

原因：未加载 Ascend/CANN 的环境变量（未 `source set_env.sh`），导致 `LD_LIBRARY_PATH` 不包含相关目录。

修复：加载 CANN/Ascend Toolkit 的 `set_env.sh` 后重试。

```bash
# 自动探测并加载 CANN 环境
CANN_ENV_SH=${CANN_ENV_SH:-$(ls -1 ${ASCEND_INSTALL_PATH:-/usr/local/Ascend}/*/set_env.sh ${ASCEND_INSTALL_PATH:-/usr/local/Ascend}/*/*/set_env.sh 2>/dev/null | head -1)}
test -n "$CANN_ENV_SH" && source "$CANN_ENV_SH" || echo "CANN not found"

npu-smi info
```

## 下载/安装包损坏：Unexpected archive size

```bash
mkdir -p /tmp/pypto_download && cd /tmp/pypto_download
rm -f Ascend-cann-*.run cann-pto-isa_*.run 2>/dev/null || true
wget --progress=bar --timeout=600 --tries=10 -O <file.run> <url>
```

## torch_npu 导入失败：共享库找不到 (libhccl.so / libatb.so / libascend_hal.so)

原因：CANN toolkit 或相关组件的 `set_env.sh` 未加载，导致 `LD_LIBRARY_PATH` 缺失。
其中，`libhccl.so` 属于 CANN toolkit 链路，`libatb.so` 属于 NNAL/ATB 组件，但症状和排查入口一致。

```bash
# 检查 CANN 是否安装
ls ${ASCEND_INSTALL_PATH:-/usr/local/Ascend}/cann-*/set_env.sh 2>/dev/null || echo 'CANN not installed'

# 加载 CANN 环境
CANN_ENV_SH=${CANN_ENV_SH:-$(ls -1 ${ASCEND_INSTALL_PATH:-/usr/local/Ascend}/*/set_env.sh ${ASCEND_INSTALL_PATH:-/usr/local/Ascend}/*/*/set_env.sh 2>/dev/null | head -1)}
test -n "$CANN_ENV_SH" && source "$CANN_ENV_SH" || echo "CANN not found"

# 若仍缺 libatb.so，额外加载 NNAL/ATB 组件
test -f "${ASCEND_INSTALL_PATH:-/usr/local/Ascend}/nnal/atb/set_env.sh" && source "${ASCEND_INSTALL_PATH:-/usr/local/Ascend}/nnal/atb/set_env.sh"

python3 -c "import torch_npu; print('torch_npu ok')"
```

若未安装，见 `references/prepare_environment.md`。

## undefined symbol / ABI 不匹配

原因：`torch` 与 `torch_npu` 版本组合不在兼容矩阵内。

修复：按官方兼容矩阵锁定版本组合重装。参考 https://github.com/Ascend/pytorch

## pypto 导入失败：DT_FP8E8M0 缺失

现象（示例）：
- `AttributeError: type object 'pypto.pypto_impl.DataType' has no attribute 'DT_FP8E8M0'`

原因：Python 侧 `pypto` 代码与已编译的 `pypto_impl` 扩展版本不一致（常见于更新了源码但复用了旧的 .so），或 Python 路径混用导致导入了另一份 `pypto_impl`。

修复：在同一份 `PYPTO_REPO` 内重新编译/重装 `pypto`。

```bash
# 0) 可选：确保 CANN 环境已加载（真实 NPU 环境）
CANN_ENV_SH=${CANN_ENV_SH:-$(ls -1 ${ASCEND_INSTALL_PATH:-/usr/local/Ascend}/*/set_env.sh ${ASCEND_INSTALL_PATH:-/usr/local/Ascend}/*/*/set_env.sh 2>/dev/null | head -1)}
test -n "$CANN_ENV_SH" && source "$CANN_ENV_SH" || echo "CANN not found"

cd "$PYPTO_REPO"

# 1) 卸载已安装的 pypto（避免旧包残留）
python3 -m pip uninstall -y pypto || true

# 2) 清理历史构建产物（避免复用旧 .so）
rm -rf build/ temp.linux-* output/ 2>/dev/null || true
rm -f python/pypto/pypto_impl*.so 2>/dev/null || true

# 3) 重新以 editable 模式安装（会触发重新构建）
export PTO_TILE_LIB_CODE_PATH=${PTO_TILE_LIB_CODE_PATH:-$PWD/pto-isa}
python3 -m pip install -e . --no-build-isolation --verbose

# 4) 验证：DT_FP8E8M0 枚举存在且 pypto 可导入
python3 -c "from pypto.pypto_impl import DataType; print('DT_FP8E8M0=', DataType.DT_FP8E8M0)"
python3 -c "import pypto; print('pypto import OK:', pypto.__file__)"
```

仍失败：优先排查是否混用多个 pypto 路径/解释器。
- `which python3`
- `python3 -c "import sys; print('\\n'.join(sys.path))"`
- `python3 -c "import pypto; print(pypto.__file__)"`

## pip 依赖冲突：ResolutionImpossible

原因：torch 版本组合冲突。以官方矩阵为准，必要时创建新 venv 重装。

## PTO ISA 编译/头文件错误

原因：`PTO_TILE_LIB_CODE_PATH` 指向错误。

```bash
export PTO_TILE_LIB_CODE_PATH="${PTO_ISA_DIR:?请先设置}"
test -d "$PTO_TILE_LIB_CODE_PATH/include/pto" && echo OK
```

## pto-isa 版本不匹配：缺少 pto::TROWEXPANDADD / pto::TROWEXPANDMAX

现象（示例）：
- `error: no member named 'TROWEXPANDADD' in namespace 'pto'`
- `error: no member named 'TROWEXPANDMAX' in namespace 'pto'`

原因：当前使用的 pto-isa 头文件版本过旧或与 PyPTO 分支不匹配（常见于使用 .run 包安装的 pto-isa，或 `PTO_TILE_LIB_CODE_PATH` 指向了错误目录）。

修复（优先源码方式）：

```bash
# 1) 切换到源码方式的 pto-isa（推荐）
PTO_ISA_DIR=${PTO_ISA_DIR:-$PWD/pto-isa}
mkdir -p "$PTO_ISA_DIR"
if [ ! -d "$PTO_ISA_DIR/.git" ]; then
  git clone https://gitcode.com/cann/pto-isa.git "$PTO_ISA_DIR"
else
  git -C "$PTO_ISA_DIR" pull --rebase
fi
export PTO_TILE_LIB_CODE_PATH="$PTO_ISA_DIR"
test -d "$PTO_TILE_LIB_CODE_PATH/include/pto" && echo OK

# 2) 清理 softmax 示例的历史编译产物（避免复用旧产物）
cd "${PYPTO_REPO:-$PWD/pypto}/examples/02_intermediate/operators/softmax"
rm -rf output/ 2>/dev/null || true

# 3) 重新跑 softmax 用例
# NPU 环境
python3 "${PYPTO_REPO:-$PWD/pypto}/examples/02_intermediate/operators/softmax/softmax.py"
# 非 NPU 环境
python3 "${PYPTO_REPO:-$PWD/pypto}/examples/02_intermediate/operators/softmax/softmax.py" --run_mode sim
```

仍失败：
- 确认 `python3 -c "import pypto; print(pypto.__file__)"` 指向的 PyPTO 与你正在使用的 `$PYPTO_REPO` 是同一份源码/同一分支（避免混用）。
- 若必须使用 .run 包安装的 pto-isa，建议改用源码方式覆盖 `PTO_TILE_LIB_CODE_PATH`（PyPTO 会优先使用该环境变量）。

## conda 中 torch 存在但 import 失败

原因：torch 安装在 conda env 中，但当前使用系统 python。

```bash
pip show torch | grep Location
which python3
```

修复：激活正确的 conda 环境。

## PYTHONPATH 导致 pypto import 异常

原因：`PYTHONPATH` 包含 pypto 源码目录的父目录（如 `/workspace/python` 或 `/workspace`），使 Python 优先找到未编译的 pypto 源码。

```bash
unset PYTHONPATH
python3 -c "import pypto; print(pypto.__file__)"
```
