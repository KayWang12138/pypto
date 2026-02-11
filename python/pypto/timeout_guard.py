#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyPTO 编译超时监控 - Python 层实现

这个模块使用 Python 的 signal.alarm() 来实现真正的编译超时控制。
可以在不修改 C++ 代码的情况下强制中断长时间运行的编译过程。

使用方式：
    from pypto.timeout_guard import TimeoutGuard, timeout_context

    # 方式1: 装饰器
    @timeout_context(timeout_sec=300, interval_sec=30)
    def run_test():
        run_compilation()

    # 方式2: 上下文管理器
    @timeout_context(timeout_sec=300)
    def run_compilation():
        pass

    # 方式3: 手动使用 TimeoutGuard
    with TimeoutGuard(timeout_sec=300):
        run_compilation()
"""

import signal
import sys
import time
import threading
import functools
import atexit
from typing import Optional, Callable


class TimeoutError(Exception):
    """编译超时异常"""
    pass


class CompilerMonitor:
    """
    编译进度监控器 - 后台线程打印进度

    只负责打印进度，不负责超时控制。
    超时控制由 TimeoutGuard 通过 signal.alarm() 实现。
    """

    def __init__(self, interval_sec: int = 60):
        """
        Args:
            interval_sec: 进度打印间隔（秒）
        """
        self.interval_sec = interval_sec
        self.start_time: Optional[float] = None
        self.current_stage = "Idle"
        self.stop_requested = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._last_print_time = 0.0
        self._stage_count = 0

    def start_stage(self, stage_name: str):
        """开始一个新的编译阶段"""
        with self._lock:
            self._stage_count += 1
            self.current_stage = f"#{self._stage_count} {stage_name}"
            if self.start_time is None:
                self.start_time = time.time()
                self._last_print_time = self.start_time

        sys.stdout.flush()
        print(f"[Compiler Monitor] [Stage Started] {self.current_stage}")
        sys.stdout.flush()

    def update_stage(self, stage_name: str):
        """更新当前阶段名称"""
        with self._lock:
            self.current_stage = f"#{self._stage_count} {stage_name}"

    def _monitor_loop(self):
        """后台监控线程循环 - 只负责打印进度"""
        while not self.stop_requested:
            time.sleep(0.5)  # 每0.5秒检查一次

            with self._lock:
                if self.start_time is None:
                    continue

                elapsed = time.time() - self.start_time

                # 定期打印进度
                if elapsed >= self.interval_sec:
                    if elapsed - self._last_print_time >= self.interval_sec:
                        elapsed_sec = int(elapsed)
                        elapsed_min = elapsed_sec // 60
                        elapsed_sec_rem = elapsed_sec % 60

                        if elapsed_min > 0:
                            elapsed_str = f"{elapsed_min}min {elapsed_sec_rem}s"
                        else:
                            elapsed_str = f"{elapsed_sec}s"

                        # 立即刷新输出
                        print(f"[Compiler Monitor] Stage: {self.current_stage} | Elapsed: {elapsed_str} ({elapsed_sec}s)")
                        sys.stdout.flush()
                        self._last_print_time = time.time()

    def start(self):
        """启动监控"""
        with self._lock:
            self.start_time = time.time()
            self._last_print_time = time.time()
            self.stop_requested = False
            self._stage_count = 0

        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="CompilerMonitorThread",
            daemon=True
        )
        self._monitor_thread.start()

        print(f"[Compiler Monitor] Started: interval={self.interval_sec}s")
        sys.stdout.flush()

    def stop(self):
        """停止监控"""
        with self._lock:
            self.stop_requested = True

        if self._monitor_thread:
            self._monitor_thread.join(timeout=1)
            self._monitor_thread = None

        if self.start_time:
            elapsed = time.time() - self.start_time
            print(f"[Compiler Monitor] Stopped: total {int(elapsed)}s")
            sys.stdout.flush()

    def get_elapsed(self) -> float:
        """获取已用时间（秒）"""
        with self._lock:
            if self.start_time:
                return time.time() - self.start_time
        return 0.0


# 全局监控实例
_global_monitor: Optional[CompilerMonitor] = None


def start_global_monitor(interval_sec: int = 60) -> CompilerMonitor:
    """启动全局监控实例"""
    global _global_monitor
    _global_monitor = CompilerMonitor(interval_sec)
    _global_monitor.start()
    return _global_monitor


def stop_global_monitor():
    """停止全局监控实例"""
    global _global_monitor
    if _global_monitor:
        _global_monitor.stop()
        _global_monitor = None


@atexit.register
def _cleanup_monitor():
    """进程退出时清理监控"""
    stop_global_monitor()


def start_stage(stage_name: str):
    """开始全局监控的新阶段"""
    global _global_monitor
    if _global_monitor:
        _global_monitor.start_stage(stage_name)


def update_stage(stage_name: str):
    """更新全局监控的当前阶段"""
    global _global_monitor
    if _global_monitor:
        _global_monitor.update_stage(stage_name)


def get_elapsed() -> float:
    """获取全局监控的已用时间"""
    global _global_monitor
    if _global_monitor:
        return _global_monitor.get_elapsed()
    return 0.0


# ==================== Timeout Guard (使用 signal.alarm) ====================

class TimeoutGuard:
    """
    超时保护器 - 使用 signal.alarm() 实现真正的超时控制

    工作原理：
    1. 进入时设置 signal.signal(SIGALRM, handler)
    2. 设置 signal.alarm(timeout_sec) 启动定时器
    3. 超时时 Python 调用 handler
    4. handler 抛出 TimeoutError
    5. 退出时取消定时器 signal.alarm(0)

    警告：这个方法不支持嵌套超时
    """

    # 类级别的标志，确保同一时间只有一个超时活跃
    _active = False
    _lock = threading.Lock()

    def __init__(self, timeout_sec: int):
        """
        Args:
            timeout_sec: 超时阈值（秒）
        """
        if timeout_sec < 1:
            raise ValueError(f"timeout_sec must be >= 1, got {timeout_sec}")

        self.timeout_sec = timeout_sec
        self._original_handler = None
        self._enabled = False

    def _timeout_handler(self, signum, frame):
        """超时信号处理函数"""
        import traceback
        elapsed = time.time() - getattr(self, '_start_time', time.time())

        msg = (
            f"\n{'=' * 60}\n"
            f"[TIMEOUT] Compilation timed out after {self.timeout_sec}s (actual: {int(elapsed)}s)\n"
            f"{'=' * 60}\n"
            f"Stack trace:\n"
        )

        # 打印当前调用栈
        traceback.print_stack(frame, file=sys.stderr)
        sys.stderr.flush()

        # 抛出超时异常（这会终止主线程中的执行）
        raise TimeoutError(
            f"Compilation timed out after {self.timeout_sec} seconds. "
            f"Actual elapsed time: {int(elapsed)} seconds."
        )

    def __enter__(self):
        """进入上下文 - 设置超时"""
        with TimeoutGuard._lock:
            if TimeoutGuard._active:
                raise RuntimeError(
                    "Nested TimeoutGuard is not supported. "
                    "Only one timeout can be active at a time."
                )

            TimeoutGuard._active = True
            self._start_time = time.time()
            self._original_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, self._timeout_handler)
            signal.alarm(self.timeout_sec)
            self._enabled = True

            print(f"[Timeout Guard] Enabled: {self.timeout_sec}s timeout")
            sys.stdout.flush()
            return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文 - 取消超时"""
        with TimeoutGuard._lock:
            if self._enabled:
                signal.alarm(0)  # 取消闹钟
                # 恢复原来的信号处理器
                if self._original_handler is not None:
                    signal.signal(signal.SIGALRM, self._original_handler)
                elif self._original_handler == 0:
                    signal.signal(signal.SIGALRM, signal.SIG_DFL)

                TimeoutGuard._active = False
                self._enabled = False

        # 如果是因为超时退出的，不抑制异常
        if isinstance(exc_val, TimeoutError):
            return False  # 重新抛出超时异常
        return False


def timeout_context(timeout_sec: int):
    """
    装饰器工厂函数 - 创建超时上下文装饰器

    Args:
        timeout_sec: 超时阈值（秒）

    Example:
        @timeout_context(timeout_sec=300)
        def run_test():
            run_compilation()

        @timeout_context(timeout_sec=120)
        def quick_test():
            run_quick_compilation()
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with TimeoutGuard(timeout_sec):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# ==================== 预设模式 ====================

class MonitorConfig:
    """监控配置预设"""

    @staticmethod
    def quick_debug(interval_sec: int = 10, timeout_sec: int = 120):
        """
        快速调试模式

        Args:
            interval_sec: 打印间隔（秒）
            timeout_sec: 超时阈值（秒）
        """
        return {
            'interval_sec': interval_sec,
            'timeout_sec': timeout_sec
        }

    @staticmethod
    def standard(interval_sec: int = 60, timeout_sec: int = 1800):
        """
        标准模式

        Args:
            interval_sec: 打印间隔（秒）
            timeout_sec: 超时阈值（秒）
        """
        return {
            'interval_sec': interval_sec,
            'timeout_sec': timeout_sec
        }

    @staticmethod
    def long_running(interval_sec: int = 300, timeout_sec: int = 7200):
        """
        长时间运行模式

        Args:
            interval_sec: 打印间隔（秒）
            timeout_sec: 超时阈值（秒）
        """
        return {
            'interval_sec': interval_sec,
            'timeout_sec': timeout_sec
        }


def run_with_monitor_and_timeout(
    func: Callable,
    interval_sec: int = 60,
    timeout_sec: int = 1800,
    timeout_action: str = "throw"
):
    """
    运行函数并启用监控和超时保护

    Args:
        func: 要运行的函数
        interval_sec: 打印间隔（秒）
        timeout_sec: 超时阈值（秒）
        timeout_action: "throw" 抛出异常, "warn" 只警告

    Returns:
        函数的返回值
    """
    if interval_sec < 1:
        raise ValueError(f"interval_sec must be >= 1, got {interval_sec}")
    if timeout_sec < 1:
        raise ValueError(f"timeout_sec must be >= 1, got {timeout_sec}")

    try:
        # 启动进度监控
        monitor = start_global_monitor(interval_sec)
        monitor.update_stage("Running")

        # 启动超时保护（如果需要）
        if timeout_action == "throw":
            with TimeoutGuard(timeout_sec):
                return func()
        else:
            # 只警告模式，不真正中断
            import warnings
            warnings.warn(f"Warning mode enabled: timeout after {timeout_sec}s")
            return func()

    finally:
        stop_global_monitor()


# ==================== 模块级便捷函数 ====================

def monitor_and_execute(
    func: Callable,
    interval_sec: int = 60,
    timeout_sec: int = 1800,
    args: tuple = (),
    kwargs: Optional[dict] = None
):
    """
    执行函数并应用监控和超时保护

    Args:
        func: 要执行的函数
        interval_sec: 监控打印间隔（秒）
        timeout_sec: 超时阈值（秒）
        args: 函数参数
        kwargs: 函数关键字参数

    Returns:
        函数的返回值

    Raises:
        TimeoutError: 如果超时
    """
    if kwargs is None:
        kwargs = {}

    def wrapped_func():
        return func(*args, **kwargs)

    return run_with_monitor_and_timeout(
        wrapped_func,
        interval_sec=interval_sec,
        timeout_sec=timeout_sec
    )


def set_monitor_mode(mode: str):
    """
    设置监控模式

    Args:
        mode: "quick", "standard", "long"

    Returns:
        dict: 配置字典
    """
    modes = {
        'quick': MonitorConfig.quick_debug,
        'standard': MonitorConfig.standard,
        'long': MonitorConfig.long_running
    }

    if mode not in modes:
        raise ValueError(f"Unknown mode: {mode}. Available: {list(modes.keys())}")

    return modes[mode]()


# ==================== 向后兼容 API ====================

def set_compiler_monitor_options(
    enable: bool = True,
    interval_sec: int = 60,
    timeout_sec: int = 1800,
    timeout_action: str = "throw",
    stage_mode: str = "coarse",
    **kwargs  # 忽略其他参数以保持兼容
):
    """
    设置编译监控选项（向后兼容）

    注意：这个Python实现会调用C++的set_compiler_monitor_options
    但超时控制完全由Python的signal.alarm()实现

    Args:
        enable: 是否启用监控
        interval_sec: 打印间隔（秒）
        timeout_sec: 超时阈值（秒）
        timeout_action: "throw" 超时抛异常, "warn" 仅警告
    """
    # 如果有C++绑定则调用
    try:
        from . import pypto_impl
        pypto_impl.SetMonitorOptions(
            enable=enable,
            interval_sec=interval_sec,
            timeout_sec=timeout_sec,
            # action 映射可能需要调整
            timeout_action=0 if timeout_action == "throw" else 1
        )
    except (ImportError, AttributeError):
        pass  # C++绑定不可用，继续使用Python实现

    # 设置超时标志供后续使用
    global _timeout_sec, _timeout_action, _monitor_enabled
    _timeout_sec = timeout_sec if enable else 0
    _timeout_action = timeout_action
    _monitor_enabled = enable
    _interval_sec_for_monitor = interval_sec if enable else 0
    return _timeout_sec, _timeout_action


_global_vars = {
    '_timeout_sec': 1800,
    '_timeout_action': 'throw',
    '_monitor_enabled': True,
    '_interval_sec_for_monitor': 60,
}


# 导出
__all__ = [
    # 主要类
    'TimeoutGuard',
    'CompilerMonitor',
    'TimeoutError',

    # 函数
    'timeout_context',
    'start_global_monitor',
    'stop_global_monitor',
    'start_stage',
    'update_stage',
    'get_elapsed',
    'run_with_monitor_and_timeout',
    'monitor_and_execute',
    'set_compiler_monitor_options',
    'set_monitor_mode',

    # 预设
    'MonitorConfig',
]


if __name__ == "__main__":
    # 测试代码
    import time

    print("=" * 60)
    print("测试 Python 层编译超时监控")
    print("=" * 60)

    # 测试1: 超时中断
    print("\n测试1: 使用 TimeoutGuard，5秒超时")
    try:
        with TimeoutGuard(timeout_sec=5):
            for i in range(15):
                print(f"  工作中... {i + 1}s")
                time.sleep(1)
        print("  成功完成")
    except TimeoutError as e:
        print(f"  ✓ 捕获超时: {e}")

    # 测试2: 进度监控
    print("\n测试2: 使用 CompilerMonitor，每2秒打印，运行8秒")
    monitor = start_global_monitor(interval_sec=2)
    monitor.start_stage("测试阶段")

    try:
        for i in range(8):
            time.sleep(1)
            print(f"  工作 {i + 1}s")
    finally:
        stop_global_monitor()

    # 测试3: 组合使用
    print("\n测试3: 组合使用监控和超时（10秒超时）")
    try:
        with TimeoutGuard(timeout_sec=10):
            monitor = start_global_monitor(interval_sec=2)
            monitor.start_stage("组合测试")

            try:
                for i in range(15):
                    time.sleep(1)
                    print(f"  工作 {i + 1}s")
            finally:
                stop_global_monitor()
        print("  成功完成")
    except TimeoutError as e:
        print(f"  ✓ 捕获超时: {e}")

    print("\n" + "=" * 60)
    print("所有测试完成")
    print("=" * 60)
