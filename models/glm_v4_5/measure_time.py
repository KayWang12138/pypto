# timer_utils.py 公共工具模块
import time

class TimeMeasurer:
    """计时工具类 - 实例属性版，支持多实例独立计数"""
    def __init__(self, repeat=5, init_index=0):
        # 初始化实例属性：计数索引、重复次数（保留你原来的repeat）
        self.index = init_index  # 替代原来的全局index
        self.repeat = repeat

    def measure_execution_time(self, func, args, desc="", *, dynamic_shape=-1):
        """
        测量函数执行耗时并打印详细信息
        参数:
            func: 要执行并计时的函数
            args: 传递给函数的参数元组（空则传()）
            desc: 执行描述（用于区分多次执行）
            dynamic_shape: 动态形状参数，默认-1不打印
        """
        # 直接使用实例属性self.index，无需global声明
        start_time = time.perf_counter()
        print(f"开始第：{self.index} 次")
        if dynamic_shape != -1:
            print(f"动态shape: {dynamic_shape}")
        if desc:
            print(desc)
        
        # 执行目标函数（兼容无参数的情况，args传()即可）
        func(*args)
        
        # 计算耗时并更新索引
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"结束第：{self.index} 次")
        print(f"耗时: {elapsed_time:.6f} 秒")
        print()
        self.index += 1  # 索引自增，保留到实例中