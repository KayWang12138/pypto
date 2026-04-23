# pmu_user_access：启用 EL0 直读 ARMv8 PMU 寄存器

本目录提供一个最小内核模块，用于开启 ARMv8 PMUv3 寄存器在 **EL0（用户态）** 的访问权限（即把 `PMUSERENR_EL0` 写成 `0xF`），让 `framework/src/machine/utils/arm_pmu_direct_sampler.h` 中的 `ArmPmuDirectSampler` 能在运行期通过 `MRS` 指令直接采样，**完全绕过 `perf_event_open` / `ioctl` / `read` 系统调用路径**。

## 什么时候需要它

- 运行时在日志里看到：
  ```
  [ARM_PMU_DIRECT] User-space PMU access disabled (PMUSERENR_EL0=0x0).
  Enable via: echo 1 > /proc/sys/kernel/perf_user_access,
  or load a kernel module that sets PMUSERENR_EL0.EN=1
  ```
- 内核版本 **< 5.17**（没有 `kernel.perf_user_access` 这个 sysctl）。如 Linux 5.10 系列。
- 容器环境下 `perf_event_open` 被 seccomp 拦截，或 `perf_event_paranoid` 被锁死，但仍希望采集 PMU 数据。

如果内核 ≥ 5.17，**不需要本模块**，直接 `sysctl -w kernel.perf_user_access=1` 即可，并在 `/etc/sysctl.d/` 下做持久化。

## 目录内容

| 文件 | 作用 |
| --- | --- |
| `pmu_user_access.c` | 内核模块源码：加载时写 `PMUSERENR_EL0 = 0xF`，并通过 `cpuhp_setup_state` 在 CPU 热插拔后保持 |
| `Makefile` | 外部模块构建脚本 |
| `probe_pmu.c` | 用户态验证程序，读取 `PMUSERENR_EL0` / `PMCCNTR_EL0` 确认权限已开 |
| `install.sh` | 一键构建 / 加载 / 验证 / 持久化 / 卸载脚本 |

## 前置依赖

- 目标机 CPU 架构 `aarch64`
- 已安装与当前运行内核完全匹配的头文件：
  - Huawei Cloud EulerOS / CentOS / openEuler：
    ```bash
    sudo dnf install -y kernel-devel-$(uname -r) kernel-headers-$(uname -r) gcc make
    ```
  - Ubuntu / Debian：
    ```bash
    sudo apt install -y linux-headers-$(uname -r) build-essential
    ```
- 已关闭 Secure Boot（或为内核模块做签名），否则 `insmod` 会被 kernel lockdown 拒绝

## 一键使用

```bash
cd tools/scripts/pmu_user_access

# 构建 + 加载 + 验证（本次启动有效）
./install.sh

# 追加持久化：拷贝到 /lib/modules/$(uname -r)/extra/，重启自动加载
./install.sh --persist

# 仅验证当前 PMU 权限状态
./install.sh --verify

# 卸载模块并清理持久化
./install.sh --uninstall
```

成功输出示例：

```
[pmu-user-access] Building pmu_user_access.ko against kernel 5.10.0-182.0.0.95.r2673_211.hce2.aarch64
[pmu-user-access] Loading pmu_user_access.ko
  [  123.456789] pmu_user_access: enabled (PMUSERENR_EL0=0xf) on all CPUs
[pmu-user-access] Probing PMU access on all 8 CPUs
  cpu00: [cpu0] PMUSERENR_EL0 = 0xf
  cpu01: [cpu1] PMUSERENR_EL0 = 0xf
  ...
[pmu-user-access] All CPUs have EL0 PMU access enabled.
```

## 手动使用（不想走脚本）

```bash
cd tools/scripts/pmu_user_access
make
sudo insmod ./pmu_user_access.ko
dmesg | tail -n 3                       # 应看到 PMUSERENR_EL0=0xf

make probe_pmu && ./probe_pmu           # 自检

# 卸载
sudo rmmod pmu_user_access
```

## 验证与 `ArmPmuDirectSampler` 对接

1. 确保目标 CMake 工程编译时带上 `-DARM_PMU_DIRECT_ENABLE=1`（这是 `arm_pmu_direct_sampler.h` 的启用开关）。
2. 模块加载成功后，原本的 warning 日志会被替换为：
   ```
   [ARM_PMU_DIRECT] Enabled, 6 hardware counters active
   ```
3. **线程亲和性**：`MRS` 读的是执行该指令的核的 PMU；如果线程在 `Begin()` 和 `End()` 之间被调度迁移到另一个核，差值会失真。对精细打点场景建议：
   ```c
   cpu_set_t set; CPU_ZERO(&set); CPU_SET(0, &set);
   pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
   ```

## 常见问题

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `insmod: ERROR: ... Operation not permitted` + dmesg 有 `Lockdown: insmod: unsigned module loading is restricted` | Secure Boot / kernel lockdown 开启 | 关闭 Secure Boot，或用内核私钥 `scripts/sign-file` 对模块签名 |
| `module verification failed: signature and/or required key missing` | 无签名的外部模块 | 一般仅 tainted 提示，可忽略，不影响功能 |
| `insmod: ERROR: ... Invalid module format` | `kernel-devel` 版本与 `uname -r` 不一致 | 重新安装匹配版本后 `make clean && make` |
| `make: *** /lib/modules/.../build: No such file or directory` | 未安装内核头文件 | 参考上文"前置依赖"安装 |
| 部分 CPU 仍显示 `PMUSERENR_EL0=0x0` | CPU 热插拔回调未生效 | 检查 `dmesg | grep pmu_user_access`；本模块已注册 `CPUHP_AP_ONLINE_DYN`，理论上不应出现 |
| 重启后失效 | 未执行 `--persist` | `./install.sh --persist` 或按手动方式配置 `/etc/modules-load.d/` |

## 内核升级后如何处理

外部模块与内核 ABI 绑定。`uname -r` 变了之后：

1. 卸载旧模块：`./install.sh --uninstall`
2. 装新内核对应的 `kernel-devel`
3. 重新 `./install.sh --persist`

若希望免维护，可将该模块封装为 DKMS 包（需 `dnf install dkms`），此处按最小实现给出，不引入 DKMS 依赖。

## 安全说明

- `PMUSERENR_EL0 = 0xF` 会让 **所有 EL0 进程** 都能读取该 CPU 上的 PMU 计数器。PMU 计数器是整核共享的微架构事件，理论上可被用于 side-channel 分析。请仅在开发 / 性能调优环境启用，**不要在生产环境长期开启**。
- 卸载模块后 `PMUSERENR_EL0` 会被清零，EL0 再执行 PMU 相关 `MRS` 指令会收到 `SIGILL`。
