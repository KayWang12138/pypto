#!/usr/bin/env bash
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# Build, load and (optionally) persist the pmu_user_access kernel module,
# which opens EL0 access to ARMv8 PMU registers so that the direct-register
# sampler in arm_pmu_direct_sampler.h can work without perf_event_open.
#
# Usage:
#   ./install.sh                   # build + load + verify (current boot only)
#   ./install.sh --persist         # also copy to /lib/modules and auto-load on boot
#   ./install.sh --uninstall       # rmmod and remove persistence config
#   ./install.sh --verify          # only run the user-space probe
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

KVER="$(uname -r)"
MARCH="$(uname -m)"
MODULE_NAME="pmu_user_access"
MODULE_KO="${MODULE_NAME}.ko"

log()  { printf '\033[1;34m[pmu-user-access]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[pmu-user-access]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[pmu-user-access]\033[0m %s\n' "$*" >&2; exit 1; }

require_aarch64() {
    [[ "${MARCH}" == "aarch64" ]] || die "Only aarch64 is supported (current: ${MARCH})"
}

require_root_or_sudo() {
    if [[ "$(id -u)" -ne 0 ]]; then
        command -v sudo >/dev/null 2>&1 || die "Need root or sudo to load kernel module"
        SUDO="sudo"
    else
        SUDO=""
    fi
}

check_kernel_headers() {
    if [[ ! -d "/lib/modules/${KVER}/build" ]]; then
        warn "Kernel headers for ${KVER} not found under /lib/modules/${KVER}/build"
        warn "Install them first. Examples:"
        warn "  dnf  install -y kernel-devel-${KVER} kernel-headers-${KVER}"
        warn "  yum  install -y kernel-devel-${KVER} kernel-headers-${KVER}"
        warn "  apt  install -y linux-headers-${KVER}"
        die  "Aborting: missing kernel build tree"
    fi
}

build_module() {
    log "Building ${MODULE_KO} against kernel ${KVER}"
    make clean >/dev/null 2>&1 || true
    make
}

load_module() {
    if lsmod | awk '{print $1}' | grep -qx "${MODULE_NAME}"; then
        log "Module already loaded, reloading"
        ${SUDO} rmmod "${MODULE_NAME}"
    fi
    log "Loading ${MODULE_KO}"
    ${SUDO} insmod "./${MODULE_KO}"
    ${SUDO} dmesg | tail -n 3 | sed 's/^/  /'
}

verify_access() {
    log "Building user-space probe"
    make probe_pmu >/dev/null

    local nproc
    nproc="$(nproc)"
    log "Probing PMU access on all ${nproc} CPUs"
    local all_ok=1
    for ((i = 0; i < nproc; i++)); do
        local out
        out="$(taskset -c "${i}" ./probe_pmu 2>&1 | head -n 1 || true)"
        printf '  cpu%02d: %s\n' "${i}" "${out}"
        if [[ "${out}" != *"0x"* ]] || [[ "${out}" == *"0x0"* ]]; then
            all_ok=0
        fi
    done
    if [[ "${all_ok}" -eq 1 ]]; then
        log "All CPUs have EL0 PMU access enabled."
    else
        warn "Some CPUs report PMUSERENR_EL0=0x0. Check dmesg for hotplug events."
    fi
}

persist_module() {
    local target_dir="/lib/modules/${KVER}/extra"
    log "Installing ${MODULE_KO} to ${target_dir}"
    ${SUDO} mkdir -p "${target_dir}"
    ${SUDO} cp "${MODULE_KO}" "${target_dir}/"
    ${SUDO} depmod -a

    log "Enabling auto-load on boot via /etc/modules-load.d/${MODULE_NAME}.conf"
    echo "${MODULE_NAME}" | ${SUDO} tee "/etc/modules-load.d/${MODULE_NAME}.conf" >/dev/null
    log "Persisted. The module will auto-load on subsequent boots of kernel ${KVER}."
    warn "Remember to rebuild+reinstall after kernel upgrade (uname -r changes)."
}

uninstall_module() {
    if lsmod | awk '{print $1}' | grep -qx "${MODULE_NAME}"; then
        log "Removing loaded module"
        ${SUDO} rmmod "${MODULE_NAME}"
    fi
    local target="/lib/modules/${KVER}/extra/${MODULE_KO}"
    if [[ -f "${target}" ]]; then
        log "Removing ${target}"
        ${SUDO} rm -f "${target}"
        ${SUDO} depmod -a
    fi
    local conf="/etc/modules-load.d/${MODULE_NAME}.conf"
    if [[ -f "${conf}" ]]; then
        log "Removing ${conf}"
        ${SUDO} rm -f "${conf}"
    fi
    log "Uninstall complete."
}

MODE="install"
PERSIST=0
for arg in "$@"; do
    case "${arg}" in
        --persist)   PERSIST=1 ;;
        --uninstall) MODE="uninstall" ;;
        --verify)    MODE="verify" ;;
        -h|--help)
            sed -n '10,16p' "$0"
            exit 0
            ;;
        *) die "Unknown argument: ${arg}" ;;
    esac
done

require_aarch64
require_root_or_sudo

case "${MODE}" in
    install)
        check_kernel_headers
        build_module
        load_module
        verify_access
        [[ "${PERSIST}" -eq 1 ]] && persist_module
        ;;
    uninstall)
        uninstall_module
        ;;
    verify)
        verify_access
        ;;
esac
