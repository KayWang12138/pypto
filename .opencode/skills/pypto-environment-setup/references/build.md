# PyPTO 编译与安装

> ⚠️ 遇到问题时重点参考 PyPTO 仓库内的官方文档：
> - `$PYPTO_REPO/docs/install/prepare_environment.md` — 环境准备
> - `$PYPTO_REPO/docs/install/build_and_install.md` — 编译与安装
## 前提条件

- 环境准备请先参考 `references/prepare_environment.md`。
- 仓库路径约定：`PYPTO_REPO="${PYPTO_REPO:-$PWD/pypto}"`。

编译工具链（可通过 `prepare_env.sh --quiet --type=deps` 一键安装）：
> 详细工具链列表及版本要求见 `$PYPTO_REPO/docs/install/prepare_environment.md` § "安装编译依赖"

Python 依赖：
```bash
cd "$PYPTO_REPO"
python3 -m pip install -r python/requirements.txt
```

## 第三方源码包

- 若网络可访问 `https://gitcode.com/cann-src-third-party`，构建时会自动下载第三方源码包。
- 若网络不可达，可手动准备第三方源码包并设置：

```bash
export PYPTO_THIRD_PARTY_PATH=<path-to-thirdparty>
```

- 或使用脚本准备：

```bash
bash tools/prepare_env.sh --quiet --type=third_party [--download-path=<path>]
```

需要的软件包：

构建时可自动从网络下载第三方源码包。若需查看缺失项和下载 URL，运行：

```bash
python3 "$SKILL_DIR/scripts/diagnose_env.py" --checklist
```

## 源码编译安装（推荐）

### 常规安装（生产环境）

```bash
cd "$PYPTO_REPO"
python3 -m pip install . --verbose
```

### 可编辑安装（开发调试）

```bash
cd "$PYPTO_REPO"
python3 -m pip install -e . --verbose
```

> 高级编译选项（Debug 编译、CMake Generator、PyPI 安装、Docker 安装等）详见：
> `$PYPTO_REPO/docs/install/build_and_install.md`
