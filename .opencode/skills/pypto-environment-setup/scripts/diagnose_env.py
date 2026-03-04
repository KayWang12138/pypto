#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
"""PyPTO/PyTorch/CANN/NPU 环境诊断。输出结构化报告供排查使用。"""
from __future__ import annotations

import argparse
import shutil
import textwrap
import glob
import json
import logging
import os
import pathlib
import platform
import re
import subprocess
import sys
from typing import Any

# 深度 NPU 检测模块（同目录，作为独立脚本运行时需 sys.path 保证）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detect_npu import (  # noqa: E402  # pyright: ignore[reportImplicitRelativeImport, reportMissingImports]
    NPUDetectionResult as _NPUDetectionResult,
)
from detect_npu import (  # noqa: E402  # pyright: ignore[reportImplicitRelativeImport, reportMissingImports]
    detect_npu as _deep_detect_npu,
)

logging.basicConfig(level=logging.INFO, format='%(message)s')


def _parse_timeout_env() -> int:
    raw = os.environ.get('DIAG_TIMEOUT', '10')
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 10
    return value if value > 0 else 10


_DEFAULT_TIMEOUT_S: int = _parse_timeout_env()


def _run(cmd: list[str], timeout_s: int = _DEFAULT_TIMEOUT_S) -> dict[str, Any]:
    try:
        p = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout_s)
        return {
            "cmd": cmd,
            "returncode": p.returncode,
            "stdout": p.stdout.strip(),
            "stderr": p.stderr.strip(),
        }
    except FileNotFoundError:
        return {"cmd": cmd, "error": "not_found"}
    except subprocess.TimeoutExpired:
        return {"cmd": cmd, "error": "timeout", "timeout_s": timeout_s}


def _safe_import(name: str) -> dict[str, Any]:
    probe_code = (
        "import importlib, json\n"
        f"name = {name!r}\n"
        "try:\n"
        "    mod = importlib.import_module(name)\n"
        "    print(json.dumps({'ok': True, 'version': getattr(mod, '__version__', None), "
        "'file': getattr(mod, '__file__', None)}, ensure_ascii=False))\n"
        "except Exception as e:\n"
        "    print(json.dumps({'ok': False, 'error': f'{type(e).__name__}: {e}'}, ensure_ascii=False))\n"
    )
    r = _run([sys.executable, '-c', probe_code], timeout_s=20)
    if r.get('returncode') != 0:
        return {'ok': False, 'error': r.get('stderr') or 'subprocess_failed'}
    stdout = r.get('stdout') or ''
    if not isinstance(stdout, str) or not stdout.strip():
        return {'ok': False, 'error': 'empty_probe_output'}
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return {'ok': False, 'error': 'invalid_probe_output'}
    if isinstance(data, dict):
        return data
    return {'ok': False, 'error': 'invalid_probe_payload'}


def _probe_torch_npu_runtime() -> dict[str, Any]:
    probe_code = (
        "import json\n"
        "try:\n"
        "    import torch\n"
        "except Exception:\n"
        "    print(json.dumps({'ok': False}))\n"
        "    raise SystemExit(0)\n"
        "out = {'ok': True, 'torch_has_npu': False, 'is_available': None, 'device_count': None}\n"
        "npu_mod = getattr(torch, 'npu', None)\n"
        "out['torch_has_npu'] = npu_mod is not None\n"
        "if npu_mod is not None:\n"
        "    try:\n"
        "        out['is_available'] = bool(npu_mod.is_available())\n"
        "    except Exception:\n"
        "        pass\n"
        "    try:\n"
        "        dev_count = npu_mod.device_count()\n"
        "        out['device_count'] = int(dev_count) if isinstance(dev_count, int) else None\n"
        "    except Exception:\n"
        "        pass\n"
        "print(json.dumps(out, ensure_ascii=False))\n"
    )
    r = _run([sys.executable, '-c', probe_code], timeout_s=20)
    if r.get('returncode') != 0:
        return {}
    stdout = r.get('stdout') or ''
    if not isinstance(stdout, str) or not stdout.strip():
        return {}
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _resolve_cann_path() -> tuple[str | None, str]:
    """从环境变量推导 CANN 版本目录和 Ascend 根路径。

    优先级链：
      1. ASCEND_HOME_PATH      — set_env.sh 导出，PyPTO 全项目使用
      2. ASCEND_TOOLKIT_HOME    — set_env.sh 导出，与 ASCEND_HOME_PATH 等价
      3. ASCEND_OPP_PATH        — set_env.sh 导出，dirname 可反推 CANN 根
      4. Fallback: 扫描 $ASCEND_INSTALL_PATH/cann-* 或 /usr/local/Ascend/cann-*

    Returns:
        (cann_version_dir, ascend_root)  —  cann_version_dir 可能为 None（需 fallback 扫描）
    """
    # 优先级 1: ASCEND_HOME_PATH
    cann_dir = os.environ.get('ASCEND_HOME_PATH')
    if cann_dir and os.path.isdir(cann_dir):
        return cann_dir, str(pathlib.Path(cann_dir).parent)

    # 优先级 2: ASCEND_TOOLKIT_HOME
    cann_dir = os.environ.get('ASCEND_TOOLKIT_HOME')
    if cann_dir and os.path.isdir(cann_dir):
        return cann_dir, str(pathlib.Path(cann_dir).parent)

    # 优先级 3: ASCEND_OPP_PATH → dirname 反推
    opp_path = os.environ.get('ASCEND_OPP_PATH')
    if opp_path and os.path.isdir(opp_path):
        candidate = str(pathlib.Path(opp_path).parent)
        if os.path.isdir(candidate):
            return candidate, str(pathlib.Path(candidate).parent)

    # Fallback: 无环境变量可用，返回 None + ascend_root
    ascend_root = os.environ.get('ASCEND_INSTALL_PATH', '/usr/local/Ascend')
    return None, ascend_root


def _detect_cann(ascend_root: str, cann_hint: str | None = None) -> dict[str, Any]:
    """检测 CANN 安装：版本、toolkit、ops。

    Args:
        ascend_root: Ascend 安装根目录（如 /usr/local/Ascend）
        cann_hint: 从环境变量推导的精确 CANN 版本目录（可选）
    """
    result: dict[str, Any] = {
        "install_path": None,
        "version": None,
        "toolkit_exists": False,
        "ops_exists": False,
        "set_env_candidates": [],
        "resolved_by": None,
    }

    cann_dir: str | None = None

    # 优先使用环境变量推导的精确路径
    if cann_hint and os.path.isdir(cann_hint):
        cann_dir = cann_hint
        result["resolved_by"] = "env"
    else:
        # Fallback: 扫描 cann-* 目录
        cann_dirs = sorted(glob.glob(os.path.join(ascend_root, "cann-*")), reverse=True)
        # 也检查 ascend-toolkit/latest 软链
        toolkit_latest = os.path.join(ascend_root, "ascend-toolkit", "latest")
        if os.path.isdir(toolkit_latest) and toolkit_latest not in cann_dirs:
            cann_dirs.insert(0, toolkit_latest)
        cann_dir = cann_dirs[0] if cann_dirs else None
        if cann_dir:
            result["resolved_by"] = "scan"

    if cann_dir:
        result["install_path"] = cann_dir
        # 版本检测：优先从 compiler/version.info 读取
        vi = os.path.join(cann_dir, "compiler", "version.info")
        if os.path.isfile(vi):
            try:
                for line in pathlib.Path(vi).read_text().splitlines():
                    m = re.match(r"Version\s*=\s*(.+)", line)
                    if m:
                        result["version"] = m.group(1).strip()
                        break
            except OSError:
                pass
        # 回退：从目录名提取
        if not result["version"]:
            m = re.search(r"cann-(\d+\.\d+\.\d+\S*)", os.path.basename(cann_dir))
            if m:
                result["version"] = m.group(1)
        # toolkit / ops
        result["toolkit_exists"] = os.path.isdir(
            os.path.join(cann_dir, "toolkit")
        ) or os.path.isfile(os.path.join(cann_dir, "set_env.sh"))
        result["ops_exists"] = os.path.isdir(os.path.join(cann_dir, "opp"))

    # set_env.sh 候选
    candidates = sorted(set(glob.glob(os.path.join(ascend_root, "*/set_env.sh")))
                        | set(glob.glob(os.path.join(ascend_root, "*/*/set_env.sh"))))
    result["set_env_candidates"] = candidates[:10]
    return result



def _detect_pypto_repo() -> dict[str, Any]:
    """自动检测 PyPTO 仓库路径。"""
    candidates = []
    
    # 从已安装的 pypto 包路径反推仓库根目录（优先级最高）
    pypto_info = _safe_import('pypto')
    if pypto_info.get('file'):
        # /path/to/pypto/python/pypto/__init__.py -> /path/to/pypto
        pypto_pkg_dir = os.path.dirname(os.path.dirname(os.path.dirname(pypto_info['file'])))
        candidates.append(pypto_pkg_dir)
    
    # 环境变量 + 常规候选路径
    _ws = os.environ.get("HOME") or os.getcwd()
    candidates.extend([
        os.environ.get("PYPTO_REPO", ""),
        os.path.join(_ws, "pypto"),
        os.path.join(os.path.expanduser("~"), "pypto"),
        os.getcwd(),
    ])
    
    for p in candidates:
        if not p:
            continue
        p = os.path.abspath(p)
        if os.path.isdir(os.path.join(p, ".git")) and os.path.isfile(os.path.join(p, "tools", "prepare_env.sh")):
            return {"path": p, "valid": True}
    return {"path": None, "valid": False}


def _parse_version(ver_str: str) -> tuple[int, ...]:
    """将版本字符串解析为可比较的 int 元组。"""
    m = re.search(r'(\d+(?:\.\d+)+)', ver_str)
    if not m:
        return (0,)
    return tuple(int(x) for x in m.group(1).split('.'))


def _probe_tool(
    name: str,
    version_args: list[str],
    version_pattern: str | None = None,
    minimum: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    path = shutil.which(name)
    info: dict[str, Any] = {
        "found": bool(path),
        "path": path,
        "version": None,
        "meets_minimum": False,
    }
    if not path:
        return info

    r = _run([name, *version_args])
    stdout = str(r.get("stdout", ""))
    if int(r.get('returncode', 0)) != 0:
        info['meets_minimum'] = False
        return info
    if version_pattern and stdout:
        m = re.search(version_pattern, stdout)
        if m:
            info["version"] = m.group(1)
            info["meets_minimum"] = minimum is None or _parse_version(m.group(1)) >= minimum
    else:
        info["meets_minimum"] = True
    return info


def _detect_build_tools() -> dict[str, Any]:
    """检测编译工具链：cmake >= 3.16.3、gcc >= 7.3.1、make、g++ >= 7.3.1、ninja、pip3、python3 >= 3.9.5。"""
    tools: dict[str, Any] = {}

    tools['cmake'] = _probe_tool('cmake', ['--version'], r'cmake version (\S+)', (3, 16, 3))
    tools['gcc'] = _probe_tool('gcc', ['--version'], r'(\d+\.\d+\.\d+)', (7, 3, 1))
    tools['make'] = _probe_tool('make', ['--version'], r'GNU Make (\S+)')
    tools['g++'] = _probe_tool('g++', ['--version'], r'(\d+\.\d+\.\d+)', (7, 3, 1))

    tools['ninja'] = _probe_tool('ninja', ['--version'], r'([0-9]+(?:\.[0-9]+)*)')

    tools['pip3'] = _probe_tool('pip3', ['--version'], r'pip (\S+)')

    # python3 版本校验 (>= 3.9.5)
    py3_path = shutil.which('python3') or sys.executable
    py3_ver = platform.python_version()
    py3_meets = _parse_version(py3_ver) >= (3, 9, 5)
    tools['python3'] = {'found': bool(py3_path), 'path': py3_path, 'version': py3_ver, 'meets_minimum': py3_meets}

    return tools


def _detect_third_party_deps(pypto_repo_path: str | None) -> dict[str, Any]:
    """检测第三方编译依赖（源码包）：nlohmann/json v3.11.3、libboundscheck v1.1.16。"""
    json_url = (
        'https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/third_party_deps/'
        'json-3.11.3.tar.gz'
    )
    securec_url = (
        'https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/third_party_deps/'
        'libboundscheck-v1.1.16.tar.gz'
    )

    result: dict[str, Any] = {
        'json': {'found': False, 'version': 'v3.11.3', 'download_url': json_url},
        'libboundscheck': {
            'found': False,
            'version': 'v1.1.16',
            'download_url': securec_url,
        },
    }

    if not pypto_repo_path:
        return result

    # 搜索路径：pypto_download/third_party_packages（prepare_env.sh 默认下载位置）、pypto/third_party
    search_dirs = [
        os.path.join(os.path.dirname(pypto_repo_path), 'pypto_download', 'third_party_packages'),
        os.path.join(pypto_repo_path, 'third_party'),
        os.path.join(pypto_repo_path, 'build', 'third_party'),
    ]

    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        try:
            entries = os.listdir(d)
        except OSError:
            continue
        for entry in entries:
            lower = entry.lower()
            if 'json' in lower and ('3.11' in lower or 'nlohmann' in lower):
                result['json']['found'] = True
            if 'boundscheck' in lower or 'securec' in lower or 'libboundscheck' in lower:
                result['libboundscheck']['found'] = True

    return result


def _detect_python_deps(pypto_repo_path: str | None) -> dict[str, Any]:
    requirements_file = None
    if pypto_repo_path:
        req_path = pathlib.Path(pypto_repo_path) / 'python' / 'requirements.txt'
        if req_path.is_file():
            requirements_file = str(req_path)

    if not requirements_file:
        return {'requirements_file': None, 'packages': [], 'missing': [], 'outdated': []}

    packages: list[dict[str, Any]] = []
    missing: list[str] = []
    outdated: list[str] = []

    try:
        lines = pathlib.Path(requirements_file).read_text(encoding='utf-8').splitlines()
    except OSError:
        return {'requirements_file': requirements_file, 'packages': [], 'missing': [], 'outdated': []}

    parsed_reqs: list[tuple[str, str | None]] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        line = line.split('#', 1)[0].strip()
        if not line:
            continue

        m = re.match(r'^([A-Za-z0-9_.-]+)\s*(>=\s*[^\s]+)?', line)
        if not m:
            continue

        pkg_name = m.group(1)
        required = m.group(2)
        if required:
            required = required.replace(' ', '')
        parsed_reqs.append((pkg_name, required))

    pkg_names = [name for name, _ in parsed_reqs]
    installed_versions: dict[str, str] = {}
    if pkg_names:
        probe_code = (
            "import json\n"
            "from importlib import metadata\n"
            f"pkgs = {pkg_names!r}\n"
            "out = {}\n"
            "for pkg in pkgs:\n"
            "    try:\n"
            "        out[pkg] = metadata.version(pkg)\n"
            "    except Exception:\n"
            "        out[pkg] = None\n"
            "print(json.dumps(out, ensure_ascii=False))\n"
        )
        timeout_s = max(20, len(pkg_names) * 2)
        r = _run([sys.executable, '-c', probe_code], timeout_s=timeout_s)
        if r.get('returncode') == 0:
            stdout = r.get('stdout') or ''
            if isinstance(stdout, str) and stdout.strip():
                try:
                    data = json.loads(stdout)
                except json.JSONDecodeError:
                    data = {}
                if isinstance(data, dict):
                    for k, v in data.items():
                        if isinstance(k, str) and isinstance(v, str):
                            installed_versions[k] = v

    for pkg_name, required in parsed_reqs:
        installed_version = installed_versions.get(pkg_name)
        installed = installed_version is not None

        status = 'ok'
        if not installed:
            status = 'missing'
            missing.append(pkg_name)
        elif required and required.startswith('>='):
            min_ver = _parse_version(required[2:])
            cur_ver = _parse_version(str(installed_version or ''))
            if cur_ver < min_ver:
                status = 'outdated'
                outdated.append(pkg_name)

        packages.append({
            'name': pkg_name,
            'required': required,
            'installed_version': installed_version,
            'status': status,
        })

    return {
        'requirements_file': requirements_file,
        'packages': packages,
        'missing': missing,
        'outdated': outdated,
    }


def _detect_npu_env(ascend_root: str) -> dict[str, Any]:
    """检测 NPU 环境：调用 detect_npu 深度检测模块，覆盖 PCI/davinci/npu-smi/ACL/torch_npu 五级探测。"""
    try:
        deep = _deep_detect_npu()
    except Exception as e:
        deep = _NPUDetectionResult()
        deep.errors.append(f'deep detect failed: {type(e).__name__}: {e}')

    # npu-smi 路径和输出（用于 commands 区域复用）
    npu_smi = shutil.which('npu-smi')
    npu_smi_ok = 'npu-smi' in deep.detection_methods
    npu_smi_output: str | None = None
    if npu_smi_ok:
        r = _run(['npu-smi', 'info'], timeout_s=15)
        if r.get('returncode') == 0 and r.get('stdout'):
            npu_smi_output = r['stdout'][:500]

    # driver / firmware 目录
    driver_dir = os.path.join(ascend_root, 'driver')
    firmware_dir = os.path.join(ascend_root, 'firmware')
    driver_exists = os.path.isdir(driver_dir)
    firmware_exists = os.path.isdir(firmware_dir)

    # 检测方式描述
    if deep.detection_methods:
        detection_method = ', '.join(deep.detection_methods)
    elif driver_exists or firmware_exists:
        detection_method = 'driver/firmware'
    else:
        detection_method = 'none'

    # device_type: 深度检测的 generation 映射（A1→a1, A2→a2, A3→a3, A3+→a3+）
    device_type = deep.device_type

    return {
        'is_npu_env': deep.npu_present or driver_exists or firmware_exists,
        'detection_method': detection_method,
        'npu_smi_found': bool(npu_smi),
        'npu_smi_ok': npu_smi_ok,
        'npu_smi_path': npu_smi,
        'npu_smi_output': npu_smi_output,
        'device_type': device_type,
        'chip_name': deep.chip_name,
        'chip_family': deep.chip_family,
        'generation': deep.generation,
        'primary_use': deep.primary_use,
        'soc_version_int': deep.soc_version_int,
        'device_count': deep.device_count,
        'driver_version': deep.driver_version,
        'detection_methods_detail': deep.detection_methods,
        'detection_errors': deep.errors,
        'driver_exists': driver_exists,
        'driver_path': driver_dir if driver_exists else None,
        'firmware_exists': firmware_exists,
        'firmware_path': firmware_dir if firmware_exists else None,
    }



def _collect_issues(
    *,
    npu_env: dict[str, Any],
    cann: dict[str, Any],
    torch_info: dict[str, Any],
    torch_npu_info: dict[str, Any],
    pypto_info: dict[str, Any],
    pto_isa: dict[str, Any],
    pypto_repo: dict[str, Any],
    build_tools: dict[str, Any],
    third_party_deps: dict[str, Any],
    python_deps: dict[str, Any],
    fast: bool = False,
) -> list[dict[str, str]]:
    """根据检测结果生成问题列表。"""
    issues: list[dict[str, str]] = []
    is_npu_env = npu_env.get('is_npu_env', False)

    # 编译工具链（--fast 模式跳过）
    if not fast:
        for name, min_ver in [
            ('cmake', '3.16.3'),
            ('gcc', '7.3.1'),
            ('make', None),
            ('g++', '7.3.1'),
            ('ninja', None),
            ('pip3', None),
            ('python3', '3.9.5'),
        ]:
            info = build_tools.get(name, {})
            if not info.get('found'):
                issues.append({'component': f'build:{name}', 'severity': 'error',
                               'message': f'{name} 未安装' + (f'（需 >= {min_ver}）' if min_ver else ''),
                               'fix_hint': 'cd $PYPTO_REPO && bash tools/prepare_env.sh --quiet --type=deps'})
            elif not info.get('meets_minimum'):
                issues.append({'component': f'build:{name}', 'severity': 'error',
                               'message': f"{name} {info.get('version')} 版本过低（需 >= {min_ver}）",
                               'fix_hint': 'cd $PYPTO_REPO && bash tools/prepare_env.sh --quiet --type=deps'})


    # 第三方编译依赖（仅当仓库有效时才报告，对非开发者无意义）（--fast 模式跳过）
    if not fast and pypto_repo.get('valid', False):
        for dep_name, dep_info in third_party_deps.items():
            if not dep_info.get('found'):
                issues.append({'component': f'third_party:{dep_name}', 'severity': 'warning',
                               'message': f"{dep_name} {dep_info.get('version', '')} 源码包未找到（编译时可自动下载）",
                               'fix_hint': 'cd $PYPTO_REPO && bash tools/prepare_env.sh --quiet --type=third_party'})


    # Python 依赖（requirements.txt）（--fast 模式跳过）
    if not fast:
        for pkg in python_deps.get('packages', []):
            if pkg['status'] == 'missing':
                issues.append({'component': f"python_dep:{pkg['name']}", 'severity': 'warning',
                               'message': (
                                   f"Python 包 {pkg['name']} 未安装"
                                   f"{(' (需 ' + pkg['required'] + ')') if pkg.get('required') else ''}"
                               ),
                               'fix_hint': 'pip3 install -r $PYPTO_REPO/python/requirements.txt'})
            elif pkg['status'] == 'outdated':
                issues.append({'component': f"python_dep:{pkg['name']}", 'severity': 'warning',
                               'message': (
                                   f"Python 包 {pkg['name']} "
                                   f"{pkg.get('installed_version', '')} 版本过低（需 {pkg['required']}）"
                               ),
                               'fix_hint': 'pip3 install -r $PYPTO_REPO/python/requirements.txt'})

    # NPU 环境 + CANN 缺失 → 需要安装 CANN
    if is_npu_env and not cann.get('install_path'):
        issues.append({'component': 'cann', 'severity': 'error',
                       'message': '检测到 NPU 环境但 CANN 未安装，需通过 prepare_env.sh --type=all 安装 CANN',
                       'fix_hint': (
                           'cd $PYPTO_REPO && bash tools/prepare_env.sh '
                           '--quiet --type=all --device-type=<a2|a3>'
                       )})
    elif not cann.get('install_path'):
        issues.append({'component': 'cann', 'severity': 'warning', 'message': 'CANN 未安装',
                       'fix_hint': '见 references/prepare_environment.md § "使用 prepare_env.sh 完整安装"'})
    elif not cann.get('set_env_candidates'):
        issues.append({'component': 'cann', 'severity': 'warning', 'message': 'CANN set_env.sh 未找到',
                       'fix_hint': 'source ${ASCEND_INSTALL_PATH:-/usr/local/Ascend}/ascend-toolkit/latest/set_env.sh'})

    if is_npu_env and not torch_info.get('ok'):
        issues.append({'component': 'torch', 'severity': 'error',
                       'message': torch_info.get('error', '无法导入 torch'),
                       'fix_hint': '见 troubleshooting.md § "torch_npu 导入失败"'})
    if is_npu_env and not torch_npu_info.get('ok'):
        issues.append({'component': 'torch_npu', 'severity': 'error',
                       'message': torch_npu_info.get('error', '无法导入 torch_npu'),
                       'fix_hint': '见 troubleshooting.md § "torch_npu 导入失败"'})

    if not pypto_info.get('ok'):
        issues.append({'component': 'pypto', 'severity': 'warning',
                       'message': pypto_info.get('error', '无法导入 pypto'),
                       'fix_hint': 'cd $PYPTO_REPO && pip install -e .'})

    pto_path = pto_isa.get('path')
    if not pto_path:
        issues.append({'component': 'pto-isa', 'severity': 'warning',
                       'message': 'PTO_TILE_LIB_CODE_PATH 未设置',
                       'fix_hint': (
                           'export PTO_TILE_LIB_CODE_PATH=$PWD/pto-isa && '
                           'git clone https://gitcode.com/cann/pto-isa.git '
                           '$PTO_TILE_LIB_CODE_PATH'
                       )})
    elif not pto_isa.get('include_pto_exists'):
        issues.append({'component': 'pto-isa', 'severity': 'warning',
                       'message': f'PTO_TILE_LIB_CODE_PATH={pto_path} 下缺少 include/pto',
                       'fix_hint': '见 troubleshooting.md § "pto-isa 版本不匹配"'})

    if not pypto_repo.get('valid'):
        issues.append({'component': 'pypto_repo', 'severity': 'warning', 'message': 'PyPTO 仓库未找到',
                       'fix_hint': 'git clone https://gitcode.com/cann/pypto.git ./pypto'})

    return issues


def _format_check_rows(data: dict) -> list[str]:  # pyright: ignore[reportMissingTypeArgument]
    rows = data.get('rows', [])
    if not rows:
        return []

    col1_w = max(len(r[0]) for r in rows) + 2
    col2_w = 6
    term_w = shutil.get_terminal_size((100, 20)).columns
    try:
        max_w = int(os.environ.get('PYPTO_DIAG_TABLE_WIDTH', '100'))
    except ValueError:
        max_w = 100
    if max_w > 0:
        term_w = min(term_w, max_w)
    detail_w = max(30, term_w - col1_w - col2_w)

    header = f"{'项目':<{col1_w}}{'状态':<{col2_w}}详情"
    lines = [header, '─' * (col1_w + col2_w + detail_w)]

    cont_prefix = ' ' * (col1_w + col2_w)

    for name, status, detail in rows:
        detail_str = '' if detail is None else str(detail)
        detail_lines = detail_str.splitlines() if detail_str else ['']
        wrapped_lines: list[str] = []
        for dl in detail_lines:
            if not dl:
                wrapped_lines.append('')
                continue
            wrapped_lines.extend(
                textwrap.wrap(
                    dl,
                    width=detail_w,
                    break_long_words=True,
                    break_on_hyphens=False,
                )
            )

        first = wrapped_lines[0] if wrapped_lines else ''
        lines.append(f'{name:<{col1_w}}{status:<{col2_w}}{first}')
        for extra in wrapped_lines[1:]:
            lines.append(f'{cont_prefix}{extra}')
    return lines


def _format_issues(issues: list[dict]) -> list[str]:  # pyright: ignore[reportMissingTypeArgument]
    lines: list[str] = []
    for i in issues:
        icon = '❌' if i['severity'] == 'error' else '⚠️'
        lines.append(f"  {icon} [{i['component']}] {i['message']}")
        if i.get('fix_hint'):
            lines.append(f"    💡 {i['fix_hint']}")
    return lines


def _format_summary(data: dict, issues: list[dict]) -> list[str]:  # pyright: ignore[reportMissingTypeArgument]
    _ = data
    errors = [i for i in issues if i['severity'] == 'error']
    warnings = [i for i in issues if i['severity'] == 'warning']
    total = len(issues)
    if total:
        return [f"\n发现 {total} 个问题（{len(errors)} 错误，{len(warnings)} 警告）："]
    return ['\n✅ 环境检测通过，未发现问题。']


def _print_checklist(report: dict[str, Any], *, fast: bool = False) -> None:
    """打印人类可读的确认清单。"""
    cann = report['cann']
    pypto_repo = report['pypto_repo']
    torch_info = report['torch']
    torch_npu_info = report['torch_npu']
    pypto_info = report['pypto']
    pto_isa = report['pto_isa']
    build_tools = report.get('build_tools', {})
    third_party = report.get('third_party_deps', {})
    python_deps = report.get('python_deps', {})
    npu_env = report.get('npu_env', {})
    issues = report.get('issues', [])
    issue_components = {i['component']: i for i in issues}

    def _status(component: str, ok: bool) -> str:
        if component in issue_components:
            sev = issue_components[component]['severity']
            return '❌' if sev == 'error' else '⚠️'
        return '✅' if ok else '⚠️'

    rows: list[tuple[str, str, str]] = []

    # ── CANN ──
    cann_path = cann.get('install_path')
    rows.append(('CANN 安装路径', _status('cann', bool(cann_path)), cann_path or '未找到'))

    cann_ver = cann.get('version', '未知')
    tk = '✅' if cann.get('toolkit_exists') else '❌'
    ops = '✅' if cann.get('ops_exists') else '❌'
    cann_detail = f"{cann_ver} (toolkit {tk}, ops {ops})" if cann_path else '未安装'
    rows.append(('CANN 版本', _status('cann', bool(cann_path)), cann_detail))

    # ── NPU 环境 ──
    is_npu = npu_env.get('is_npu_env', False)
    if not is_npu:
        npu_detail = '未检测到 NPU 环境'
    else:
        if npu_env.get('npu_smi_ok'):
            npu_smi_state = 'ok'
        elif npu_env.get('npu_smi_found'):
            npu_smi_state = 'fallback'
        else:
            npu_smi_state = 'missing'

        driver_state = '✅' if npu_env.get('driver_exists') else '⚠️'
        firmware_state = '✅' if npu_env.get('firmware_exists') else '⚠️'

        det_methods = npu_env.get('detection_methods_detail', [])
        summary_line = f"npu-smi: {npu_smi_state}; driver: {driver_state}; firmware: {firmware_state}"
        if det_methods:
            npu_detail = "\n".join([summary_line, f"methods: {', '.join(det_methods)}"])
        else:
            npu_detail = summary_line
    rows.append(('NPU 环境', '✅' if is_npu else '⚠️', npu_detail))

    # ── NPU 芯片型号 ──
    dt = npu_env.get('device_type')
    cn = npu_env.get('chip_name')
    gen = npu_env.get('generation')
    dev_count = npu_env.get('device_count', 0)

    display_gen = gen
    if not display_gen and isinstance(dt, str) and dt:
        display_gen = dt.upper()

    if not is_npu:
        chip_status = '─'
        chip_detail = '不适用（非 NPU 环境）'
    else:
        chip_status = '✅' if (display_gen or cn) else '⚠️'
        if display_gen and cn:
            parts = [cn]
            if dev_count > 0:
                parts.append(f'{dev_count}设备')
            chip_detail = f"{display_gen} ({', '.join(parts)})"
        elif display_gen:
            chip_detail = display_gen
        elif cn:
            chip_detail = cn
        else:
            chip_detail = '未能识别芯片型号'

    rows.append(('芯片型号', chip_status, chip_detail))

    # ── Python ──
    py = report['python']
    rows.append(('Python 版本', '✅', f"{py['version']} ({py['executable']})"))

    # ── torch ──
    if torch_info.get('ok'):
        rows.append(('torch', '✅', str(torch_info.get('version'))))
    else:
        rows.append(('torch', _status('torch', False), torch_info.get('error', '未安装')))

    # ── torch_npu ──
    if torch_npu_info.get('ok'):
        rows.append(('torch_npu', '✅', str(torch_npu_info.get('version'))))
    else:
        rows.append(('torch_npu', _status('torch_npu', False), torch_npu_info.get('error', '未安装')))

    # ── pypto ──
    if pypto_info.get('ok'):
        detail = str(pypto_info.get('version') or pypto_info.get('file'))
        rows.append(('pypto', '✅', detail))
    else:
        rows.append(('pypto', _status('pypto', False), pypto_info.get('error', '未安装')))

    # ── PyPTO 仓库 ──
    repo_path = pypto_repo.get('path')
    rows.append(('PyPTO 仓库路径', _status('pypto_repo', bool(repo_path)), repo_path or '未找到'))

    # ── pto-isa ──
    pto_path = pto_isa.get('path')
    if pto_path and pto_isa.get('include_pto_exists'):
        rows.append(('pto-isa', '✅', pto_path))
    else:
        rows.append(('pto-isa', _status('pto-isa', False), pto_path or 'PTO_TILE_LIB_CODE_PATH 未设置'))

    # ── 编译工具链（--fast 模式跳过）──
    if not fast:
        for name, min_ver in [
            ('cmake', '>= 3.16.3'),
            ('gcc', '>= 7.3.1'),
            ('make', ''),
            ('g++', '>= 7.3.1'),
            ('ninja', ''),
            ('pip3', ''),
            ('python3', '>= 3.9.5'),
        ]:
            info = build_tools.get(name, {})
            found = info.get('found', False)
            ver = info.get('version', '未知')
            meets = info.get('meets_minimum', False)
            detail = f"{ver} ({info.get('path', '')})" if found else f"未安装 {('（需 ' + min_ver + '）') if min_ver else ''}"
            rows.append((name, _status(f'build:{name}', found and meets), detail))

    # ── 第三方编译依赖（仅当 pypto 仓库有效时显示，对非开发者无意义）──
    if pypto_repo.get('valid'):
        for dep_name, dep_info in third_party.items():
            found = dep_info.get('found', False)
            ver = dep_info.get('version', '')
            url = dep_info.get('download_url', '')
            if found:
                detail = f"{ver} ✅"
            else:
                detail = f"{ver} 未找到（编译时可自动下载）"
                if url:
                    detail += f"\n下载: {url}"
            rows.append((dep_name, '✅' if found else '⚠️', detail))

    # ── Python 依赖（requirements.txt）（--fast 模式跳过）──
    if not fast:
        pkgs = python_deps.get('packages', [])
        if pkgs:
            missing_count = len(python_deps.get('missing', []))
            outdated_count = len(python_deps.get('outdated', []))
            total = len(pkgs)
            ok_count = total - missing_count - outdated_count
            summary = f"{ok_count}/{total} 已安装"
            if missing_count:
                summary += f"，{missing_count} 缺失"
            if outdated_count:
                summary += f"，{outdated_count} 版本过低"
            overall_status = '✅' if missing_count == 0 and outdated_count == 0 else '⚠️'
            rows.append(('Python 依赖', overall_status, summary))
            # Individual packages only if there are problems
            for pkg in pkgs:
                if pkg['status'] != 'ok':
                    comp = f"python_dep:{pkg['name']}"
                    if pkg['status'] == 'missing':
                        rows.append(
                            (
                                f"  {pkg['name']}",
                                _status(comp, False),
                                (
                                    f"未安装"
                                    f"{(' (需 ' + pkg['required'] + ')') if pkg.get('required') else ''}"
                                ),
                            )
                        )
                    elif pkg['status'] == 'outdated':
                        rows.append(
                            (
                                f"  {pkg['name']}",
                                _status(comp, False),
                                f"{pkg.get('installed_version', '?')} → 需 {pkg['required']}",
                            )
                        )
        elif python_deps.get('requirements_file') is None:
            rows.append(('Python 依赖', '─', 'requirements.txt 未找到'))

    # ── 输出 ──
    logging.info('=== PyPTO 环境检测结果 ===\n')
    for line in _format_check_rows({'rows': rows}):
        logging.info(line)
    for line in _format_summary(report, issues):
        logging.info(line)
    for line in _format_issues(issues):
        logging.info(line)


def main() -> int:
    ap = argparse.ArgumentParser(description='Diagnose PyPTO/PyTorch/CANN/NPU environment')
    ap.add_argument('--json', action='store_true', help='JSON output')
    ap.add_argument('--pretty', action='store_true', help='Pretty-print JSON')
    ap.add_argument('--checklist', action='store_true', help='Human-readable confirmation checklist')
    ap.add_argument(
        '--fast',
        action='store_true',
        help='Fast mode: skip slow checks (build tools, third-party deps, python deps)',
    )
    args = ap.parse_args()

    # 环境变量（仅 Ascend 相关条目）
    env: dict[str, Any] = {}
    for k in (
        'ASCEND_HOME_PATH',
        'ASCEND_TOOLKIT_HOME',
        'ASCEND_OPP_PATH',
        'LD_LIBRARY_PATH',
        'PATH',
        'PTO_TILE_LIB_CODE_PATH',
        'PYTHONPATH',
    ):
        raw = os.environ.get(k)
        if raw is None:
            env[k] = None
        elif k in ('PATH', 'LD_LIBRARY_PATH'):
            entries = [p for p in raw.split(os.pathsep) if 'ascend' in p.lower() or 'cann' in p.lower()]
            env[k] = entries if entries else '(set but no Ascend entries)'
        else:
            env[k] = raw

    torch_info = _safe_import('torch')
    torch_npu_info = _safe_import('torch_npu')
    pypto_info = _safe_import('pypto')

    # pto-isa 目录验证
    pto_isa_path = os.environ.get('PTO_TILE_LIB_CODE_PATH')
    pto_isa: dict[str, Any] = {'path': pto_isa_path}
    if pto_isa_path:
        pto_isa['include_pto_exists'] = os.path.isdir(os.path.join(pto_isa_path, 'include', 'pto'))

    # CANN 路径推导（优先环境变量，fallback 目录扫描）
    cann_hint, ascend_root = _resolve_cann_path()

    # CANN 检测
    cann = _detect_cann(ascend_root, cann_hint=cann_hint)

    # PyPTO 仓库检测
    pypto_repo = _detect_pypto_repo()

    # 编译工具链检测（--fast 模式跳过）
    build_tools = {} if args.fast else _detect_build_tools()

    # 第三方编译依赖检测（--fast 模式跳过）
    third_party_deps = {} if args.fast else _detect_third_party_deps(pypto_repo.get('path'))

    # Python 依赖检测（requirements.txt）（--fast 模式跳过）
    python_deps = (
        {'requirements_file': None, 'packages': [], 'missing': [], 'outdated': []}
        if args.fast
        else _detect_python_deps(pypto_repo.get('path'))
    )

    # NPU 环境检测（优先 npu-smi info，fallback driver/firmware）
    npu_env = _detect_npu_env(ascend_root)

    # NPU torch 探测（详细信息）
    npu: dict[str, Any] = {}
    npu_smi = npu_env.get('npu_smi_path')
    # npu-smi info 已在 _detect_npu_env 中执行，复用结果
    commands: dict[str, Any] = {
        'npu-smi': {'found': npu_env['npu_smi_found'], 'path': npu_smi},
        'npu-smi info': {'ok': npu_env.get('npu_smi_ok', False), 'output_preview': npu_env.get('npu_smi_output')},
    }

    if torch_info.get('ok'):
        npu = _probe_torch_npu_runtime()

    # 问题检测
    issues = _collect_issues(
        npu_env=npu_env,
        cann=cann,
        torch_info=torch_info,
        torch_npu_info=torch_npu_info,
        pypto_info=pypto_info,
        pto_isa=pto_isa,
        pypto_repo=pypto_repo,
        build_tools=build_tools,
        third_party_deps=third_party_deps,
        python_deps=python_deps,
        fast=args.fast,
    )

    report = {
        'python': {'executable': sys.executable, 'version': platform.python_version()},
        'platform': {'system': platform.system(), 'machine': platform.machine()},
        'env': env,
        'build_tools': build_tools,
        'third_party_deps': third_party_deps,
        'python_deps': python_deps,
        'npu_env': npu_env,
        'cann': cann,
        'torch': torch_info,
        'torch_npu': torch_npu_info,
        'pypto': pypto_info,
        'pypto_repo': pypto_repo,
        'pto_isa': pto_isa,
        'commands': commands,
        'npu': npu,
        'issues': issues,
    }

    if args.checklist:
        _print_checklist(report, fast=args.fast)
    elif args.json or args.pretty:
        logging.info(
            json.dumps(
                report,
                ensure_ascii=False,
                indent=2 if args.pretty else None,
                sort_keys=True,
            )
        )
    else:
        logging.info(
            f"cann: {cann.get('version') or 'not found'} (path={cann.get('install_path')})"
        )
        method = npu_env.get('detection_method', 'none')
        npu_env_line = (
            f"npu_env: {'yes' if npu_env.get('is_npu_env') else 'no'} "
            f"(method={method}, "
            f"npu-smi={'ok' if npu_env.get('npu_smi_ok') else 'fail' if npu_env.get('npu_smi_found') else 'missing'}, "
            f"driver={'yes' if npu_env.get('driver_exists') else 'no'}, "
            f"firmware={'yes' if npu_env.get('firmware_exists') else 'no'})"
        )
        logging.info(npu_env_line)
        # 芯片信息（深度检测）
        cn = npu_env.get('chip_name')
        dt = npu_env.get('device_type')
        family = npu_env.get('chip_family', 'unknown')
        gen = npu_env.get('generation', 'unknown')
        soc_int = npu_env.get('soc_version_int')
        dev_count = npu_env.get('device_count', 0)
        logging.info(
            f"npu_chip: {cn or 'unknown'} "
            f"(family={family}, gen={gen}, soc={soc_int}, devices={dev_count})"
        )
        logging.info(f"device_type: {dt or 'unknown'}")
        logging.info(f"npu-smi: {'found' if npu_smi else 'missing'}")
        logging.info(f"torch.npu.is_available: {npu.get('is_available')}")
        logging.info(f"torch.npu.device_count: {npu.get('device_count')}")
        # 其他
        logging.info(f"python: {platform.python_version()} ({sys.executable})")
        logging.info(
            f"torch: {torch_info.get('version') if torch_info.get('ok') else torch_info.get('error')}"
        )
        logging.info(
            f"torch_npu: {torch_npu_info.get('version') if torch_npu_info.get('ok') else torch_npu_info.get('error')}"
        )
        pypto_status = (
            pypto_info.get('version') or pypto_info.get('file')
            if pypto_info.get('ok')
            else pypto_info.get('error')
        )
        logging.info(f"pypto: {pypto_status}")
        logging.info(
            f"pypto_repo: {pypto_repo.get('path') or 'not found'} "
            f"({'valid' if pypto_repo.get('valid') else 'invalid'})"
        )
        logging.info(
            f"pto-isa: {pto_isa_path or 'not set'} "
            f"({'valid' if pto_isa.get('include_pto_exists') else 'invalid/missing'})"
        )
        # 编译工具链（--fast 模式跳过）
        if not args.fast:
            for name in ('cmake', 'gcc', 'make', 'g++', 'ninja', 'pip3', 'python3'):
                info = build_tools.get(name, {})
                ver = info.get('version', 'missing') if info.get('found') else 'missing'
                ok = '✓' if info.get('meets_minimum') else '✗'
                logging.info(f"{name}: {ver} ({ok})")
        # 第三方依赖（--fast 模式跳过）
        if not args.fast:
            for dep_name, dep_info in third_party_deps.items():
                logging.info(
                    f"{dep_name}: {'found' if dep_info.get('found') else 'not found'} "
                    f"({dep_info.get('version', '')})"
                )
        # Python 依赖（--fast 模式跳过）
        if not args.fast:
            missing_raw = python_deps.get('missing', [])
            outdated_raw = python_deps.get('outdated', [])
            packages_raw = python_deps.get('packages', [])
            missing = missing_raw if isinstance(missing_raw, list) else []
            outdated = outdated_raw if isinstance(outdated_raw, list) else []
            total_pkgs = len(packages_raw) if isinstance(packages_raw, list) else 0
            logging.info(
                f"python_deps: {total_pkgs - len(missing) - len(outdated)}/{total_pkgs} ok "
                f"({len(missing)} missing, {len(outdated)} outdated)"
            )
        if issues:
            logging.info(f"\nissues ({len(issues)}):")
            for i in issues:
                icon = 'ERR' if i['severity'] == 'error' else 'WARN'
                logging.info(f"  [{icon}] {i['component']}: {i['message']}")

    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except BrokenPipeError as e:
        raise SystemExit(0) from e
