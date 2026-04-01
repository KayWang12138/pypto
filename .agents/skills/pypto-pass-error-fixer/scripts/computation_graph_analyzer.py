#!/usr/bin/env python3
"""
PyPTO 计算图 JSON 分析工具

本工具提供完整的计算图分析能力，包括：
- 加载和解析计算图JSON文件
- 获取各类数据（Tensor、Operation、RawTensor、Incast/Outcast等）
- 追踪数据流转
- 对比Pass前后差异
- 识别性能瓶颈
- 分析子图关系
- 分析内存分配
"""

import json
import os
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum


class GraphType(Enum):
    """计算图类型"""
    TENSOR_GRAPH = 1
    TILE_GRAPH = 2
    BLOCK_GRAPH = 3
    EXECUTE_GRAPH = 4
    
    @classmethod
    def from_value(cls, value: int) -> Optional['GraphType']:
        return cls(value) if value in cls._value2member_map_ else None
    
    @classmethod
    def to_string(cls, value: int) -> str:
        mapping = {
            1: 'TensorGraph',
            2: 'TileGraph',
            3: 'BlockGraph',
            4: 'ExecuteGraph'
        }
        return mapping.get(value, 'Unknown')


class MemoryType(Enum):
    """内存层级类型"""
    MEM_UB = 0
    MEM_L1 = 1
    MEM_L0A = 2
    MEM_L0B = 3
    MEM_L0C = 4
    MEM_DEVICE_DDR = 15
    
    @classmethod
    def to_string(cls, value: int) -> str:
        mapping = {
            0: 'MEM_UB (Unified Buffer)',
            1: 'MEM_L1 (L1 Cache)',
            2: 'MEM_L0A (L0A Cache)',
            3: 'MEM_L0B (L0B Cache)',
            4: 'MEM_L0C (L0C Cache)',
            15: 'MEM_DEVICE_DDR (Global Memory)'
        }
        return mapping.get(value, f'Unknown ({value})')


@dataclass
class TensorInfo:
    """Tensor信息"""
    magic: int
    shape: List[int]
    validshape: List[int]
    dynvalidshape: List[List[int]]
    offset: List[int]
    rawtensor: int
    nodetype: int
    mem_id: int
    mem_range: List[int]
    mem_type_asis: int
    mem_type_tobe: int
    subgraphid: int
    subgraph_boundary: bool
    life_range: List[int]
    kind: int
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    def get_memory_type_str(self) -> Tuple[str, str]:
        """获取内存类型字符串"""
        return (MemoryType.to_string(self.mem_type_asis), 
                MemoryType.to_string(self.mem_type_tobe))
    
    def is_input(self) -> bool:
        """是否为输入Tensor"""
        return self.nodetype == 1
    
    def is_output(self) -> bool:
        """是否为输出Tensor"""
        return self.nodetype == 2


@dataclass
class OperationInfo:
    """Operation信息"""
    opmagic: int
    opcode: str
    ioperands: List[int]
    ooperands: List[int]
    attr: List[Any]
    subgraphid: int
    latency: int
    kind: int
    tile: Optional[Dict[str, Any]]
    op_attr: Dict[str, Any]
    file: Optional[str]
    line: Optional[int]
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    def get_input_tensor_magics(self) -> List[int]:
        """获取输入Tensor的magic ID列表"""
        return self.ioperands
    
    def get_output_tensor_magics(self) -> List[int]:
        """获取输出Tensor的magic ID列表"""
        return self.ooperands


@dataclass
class RawTensorInfo:
    """RawTensor信息"""
    rawmagic: int
    rawshape: List[int]
    ori_rawshape: List[int]
    datatype: int
    format: int
    kind: int
    symbol: Optional[str]
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FunctionInfo:
    """Function信息"""
    func_magicname: str
    funcmagic: int
    graphtype: int
    incasts: List[List[Any]]
    outcasts: List[List[Any]]
    operations: List[OperationInfo]
    tensors: List[TensorInfo]
    rawtensors: List[RawTensorInfo]
    total_subgraph_count: int
    file: Optional[str]
    line: Optional[int]
    global_tensors: List[int]
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    def get_graph_type(self) -> Optional[GraphType]:
        """获取图类型"""
        return GraphType.from_value(self.graphtype)
    
    def get_graph_type_str(self) -> str:
        """获取图类型字符串"""
        return GraphType.to_string(self.graphtype)
    
    def get_input_tensor_magics(self) -> List[int]:
        """获取输入Tensor的magic ID列表"""
        return [incast[0] for incast in self.incasts]
    
    def get_output_tensor_magics(self) -> List[int]:
        """获取输出Tensor的magic ID列表"""
        return [outcast[0] for outcast in self.outcasts]


@dataclass
class GraphInfo:
    """计算图信息"""
    entryhash: str
    version: str
    functions: List[FunctionInfo]
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    def get_main_function(self) -> Optional[FunctionInfo]:
        """获取主函数（通常只有一个）"""
        return self.functions[0] if self.functions else None


class ComputationGraphAnalyzer:
    """计算图分析器"""
    
    def __init__(self):
        self.graph: Optional[GraphInfo] = None
    
    def load_graph(self, json_path: str) -> GraphInfo:
        """加载计算图JSON文件"""
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"计算图文件不存在: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.graph = self._parse_graph(data)
        return self.graph
    
    def _parse_graph(self, data: Dict[str, Any]) -> GraphInfo:
        """解析计算图数据"""
        functions = []
        for func_data in data.get('functions', []):
            functions.append(self._parse_function(func_data))
        
        return GraphInfo(
            entryhash=data.get('entryhash', ''),
            version=data.get('version', ''),
            functions=functions,
            raw_data=data
        )
    
    def _parse_function(self, func_data: Dict[str, Any]) -> FunctionInfo:
        """解析Function数据"""
        operations = []
        for op_data in func_data.get('operations', []):
            operations.append(self._parse_operation(op_data))
        
        tensors = []
        for tensor_data in func_data.get('tensors', []):
            tensors.append(self._parse_tensor(tensor_data))
        
        rawtensors = []
        for rt_data in func_data.get('rawtensors', []):
            rawtensors.append(self._parse_rawtensor(rt_data))
        
        return FunctionInfo(
            func_magicname=func_data.get('func_magicname', ''),
            funcmagic=func_data.get('funcmagic', -1),
            graphtype=func_data.get('graphtype', -1),
            incasts=func_data.get('incasts', []),
            outcasts=func_data.get('outcasts', []),
            operations=operations,
            tensors=tensors,
            rawtensors=rawtensors,
            total_subgraph_count=func_data.get('_total_subgraph_count', 0),
            file=func_data.get('file'),
            line=func_data.get('line'),
            global_tensors=func_data.get('global_tensors', []),
            raw_data=func_data
        )
    
    def _parse_operation(self, op_data: Dict[str, Any]) -> OperationInfo:
        """解析Operation数据"""
        return OperationInfo(
            opmagic=op_data.get('opmagic', -1),
            opcode=op_data.get('opcode', ''),
            ioperands=op_data.get('ioperands', []),
            ooperands=op_data.get('ooperands', []),
            attr=op_data.get('attr', []),
            subgraphid=op_data.get('subgraphid', -1),
            latency=op_data.get('latency', 0),
            kind=op_data.get('kind', -1),
            tile=op_data.get('tile'),
            op_attr=op_data.get('op_attr', {}),
            file=op_data.get('file'),
            line=op_data.get('line'),
            raw_data=op_data
        )
    
    def _parse_tensor(self, tensor_data: Dict[str, Any]) -> TensorInfo:
        """解析Tensor数据"""
        mem_type = tensor_data.get('mem_type', {})
        return TensorInfo(
            magic=tensor_data.get('magic', -1),
            shape=tensor_data.get('shape', []),
            validshape=tensor_data.get('validshape', []),
            dynvalidshape=tensor_data.get('dynvalidshape', []),
            offset=tensor_data.get('offset', []),
            rawtensor=tensor_data.get('rawtensor', -1),
            nodetype=tensor_data.get('nodetype', 0),
            mem_id=tensor_data.get('mem_id', -1),
            mem_range=tensor_data.get('mem_range', []),
            mem_type_asis=mem_type.get('asis', 0),
            mem_type_tobe=mem_type.get('tobe', 0),
            subgraphid=tensor_data.get('subgraphid', -1),
            subgraph_boundary=tensor_data.get('subgraph_boundary', False),
            life_range=tensor_data.get('life_range', []),
            kind=tensor_data.get('kind', -1),
            raw_data=tensor_data
        )
    
    def _parse_rawtensor(self, rt_data: Dict[str, Any]) -> RawTensorInfo:
        """解析RawTensor数据"""
        return RawTensorInfo(
            rawmagic=rt_data.get('rawmagic', -1),
            rawshape=rt_data.get('rawshape', []),
            ori_rawshape=rt_data.get('ori_rawshape', []),
            datatype=rt_data.get('datatype', -1),
            format=rt_data.get('format', -1),
            kind=rt_data.get('kind', -1),
            symbol=rt_data.get('symbol'),
            raw_data=rt_data
        )
    
    def find_tensor_by_magic(self, magic: int) -> Optional[TensorInfo]:
        """根据magic ID查找Tensor"""
        if not self.graph:
            return None
        
        func = self.graph.get_main_function()
        if not func:
            return None
        
        for tensor in func.tensors:
            if tensor.magic == magic:
                return tensor
        return None
    
    def find_operation_by_magic(self, opmagic: int) -> Optional[OperationInfo]:
        """根据opmagic ID查找Operation"""
        if not self.graph:
            return None
        
        func = self.graph.get_main_function()
        if not func:
            return None
        
        for op in func.operations:
            if op.opmagic == opmagic:
                return op
        return None
    
    def find_operations_by_opcode(self, opcode: str) -> List[OperationInfo]:
        """根据opcode查找所有Operation"""
        if not self.graph:
            return []
        
        func = self.graph.get_main_function()
        if not func:
            return []
        
        return [op for op in func.operations if opcode in op.opcode]
    
    def find_operations_by_input_tensor(self, tensor_magic: int) -> List[OperationInfo]:
        """查找使用指定Tensor作为输入的所有Operation"""
        if not self.graph:
            return []
        
        func = self.graph.get_main_function()
        if not func:
            return []
        
        return [op for op in func.operations if tensor_magic in op.ioperands]
    
    def find_operations_by_output_tensor(self, tensor_magic: int) -> List[OperationInfo]:
        """查找产生指定Tensor作为输出的所有Operation"""
        if not self.graph:
            return []
        
        func = self.graph.get_main_function()
        if not func:
            return []
        
        return [op for op in func.operations if tensor_magic in op.ooperands]
    
    def find_producer_of_tensor(self, tensor_magic: int) -> Optional[OperationInfo]:
        """查找产生指定Tensor的Operation（生产者）"""
        ops = self.find_operations_by_output_tensor(tensor_magic)
        return ops[0] if ops else None
    
    def find_consumers_of_tensor(self, tensor_magic: int) -> List[OperationInfo]:
        """查找使用指定Tensor的所有Operation（消费者）"""
        return self.find_operations_by_input_tensor(tensor_magic)
    
    def trace_data_flow(self, start_magic: int, max_depth: int = 100) -> List[int]:
        """追踪从给定Tensor开始的数据流
        
        Args:
            start_magic: 起始Tensor的magic ID
            max_depth: 最大追踪深度
            
        Returns:
            数据流路径上的Tensor magic ID列表
        """
        if not self.graph:
            return []
        
        func = self.graph.get_main_function()
        if not func:
            return []
        
        flow = [start_magic]
        visited = set([start_magic])
        current = start_magic
        depth = 0
        
        while depth < max_depth:
            # 找到以current为输入的operation
            found = False
            for op in func.operations:
                if current in op.ioperands:
                    for output in op.ooperands:
                        if output not in visited:
                            flow.append(output)
                            visited.add(output)
                            current = output
                            found = True
                            break
                    if found:
                        break
            if not found:
                break
            depth += 1
        
        return flow
    
    def trace_data_flow_backward(self, start_magic: int, max_depth: int = 100) -> List[int]:
        """反向追踪数据流（从输出到输入）
        
        Args:
            start_magic: 起始Tensor的magic ID
            max_depth: 最大追踪深度
            
        Returns:
            反向数据流路径上的Tensor magic ID列表
        """
        if not self.graph:
            return []
        
        func = self.graph.get_main_function()
        if not func:
            return []
        
        flow = [start_magic]
        visited = set([start_magic])
        current = start_magic
        depth = 0
        
        while depth < max_depth:
            # 找到产生current的operation
            producer = self.find_producer_of_tensor(current)
            if producer:
                for input_magic in producer.ioperands:
                    if input_magic not in visited:
                        flow.append(input_magic)
                        visited.add(input_magic)
                        current = input_magic
                        break
                depth += 1
            else:
                break
        
        return flow
    
    def count_nodes(self) -> Dict[str, int]:
        """统计节点数量"""
        if not self.graph:
            return {}
        
        func = self.graph.get_main_function()
        if not func:
            return {}
        
        return {
            'operations': len(func.operations),
            'tensors': len(func.tensors),
            'rawtensors': len(func.rawtensors),
            'subgraphs': func.total_subgraph_count
        }
    
    def get_operations_by_type(self) -> Dict[str, int]:
        """按类型统计Operation数量"""
        if not self.graph:
            return {}
        
        func = self.graph.get_main_function()
        if not func:
            return {}
        
        type_count = {}
        for op in func.operations:
            op_type = op.opcode
            type_count[op_type] = type_count.get(op_type, 0) + 1
        
        return type_count
    
    def get_subgraph_operations(self, subgraph_id: int) -> List[OperationInfo]:
        """获取指定子图的所有Operation"""
        if not self.graph:
            return []
        
        func = self.graph.get_main_function()
        if not func:
            return []
        
        return [op for op in func.operations if op.subgraphid == subgraph_id]
    
    def get_subgraph_tensors(self, subgraph_id: int) -> List[TensorInfo]:
        """获取指定子图的所有Tensor"""
        if not self.graph:
            return []
        
        func = self.graph.get_main_function()
        if not func:
            return []
        
        return [tensor for tensor in func.tensors if tensor.subgraphid == subgraph_id]
    
    def analyze_memory_usage(self) -> Dict[str, Any]:
        """分析内存使用情况"""
        if not self.graph:
            return {}
        
        func = self.graph.get_main_function()
        if not func:
            return {}
        
        memory_stats = {
            'allocated_tensors': 0,
            'unallocated_tensors': 0,
            'memory_by_type': {},
            'total_memory_range': 0
        }
        
        for tensor in func.tensors:
            if tensor.mem_id != -1:
                memory_stats['allocated_tensors'] += 1
                mem_type_key = f"{tensor.mem_type_asis}->{tensor.mem_type_tobe}"
                memory_stats['memory_by_type'][mem_type_key] = \
                    memory_stats['memory_by_type'].get(mem_type_key, 0) + 1
            else:
                memory_stats['unallocated_tensors'] += 1
        
        return memory_stats
    
    def find_tensor_with_multiple_consumers(self, opcode_filter: Optional[str] = None) -> List[Tuple[TensorInfo, List[OperationInfo]]]:
        """查找有多个消费者的Tensor
        
        Args:
            opcode_filter: 可选，只统计指定类型的消费者
            
        Returns:
            [(Tensor, [Consumer1, Consumer2, ...]), ...]
        """
        if not self.graph:
            return []
        
        func = self.graph.get_main_function()
        if not func:
            return []
        
        results = []
        for tensor in func.tensors:
            consumers = self.find_consumers_of_tensor(tensor.magic)
            
            if opcode_filter:
                consumers = [op for op in consumers if opcode_filter in op.opcode]
            
            if len(consumers) > 1:
                results.append((tensor, consumers))
        
        return results
    
    def get_summary(self) -> Dict[str, Any]:
        """获取计算图摘要信息"""
        if not self.graph:
            return {}
        
        func = self.graph.get_main_function()
        if not func:
            return {}
        
        return {
            'entryhash': self.graph.entryhash,
            'version': self.graph.version,
            'function_name': func.func_magicname,
            'function_magic': func.funcmagic,
            'graph_type': func.get_graph_type_str(),
            'node_counts': self.count_nodes(),
            'operation_types': self.get_operations_by_type(),
            'input_tensors': func.get_input_tensor_magics(),
            'output_tensors': func.get_output_tensor_magics(),
            'subgraph_count': func.total_subgraph_count,
            'memory_stats': self.analyze_memory_usage()
        }


def compare_graphs(before_analyzer: ComputationGraphAnalyzer, 
                   after_analyzer: ComputationGraphAnalyzer) -> Dict[str, Any]:
    """对比两个计算图
    
    Args:
        before_analyzer: 修改前的计算图分析器
        after_analyzer: 修改后的计算图分析器
        
    Returns:
        对比结果字典
    """
    before_func = before_analyzer.graph.get_main_function() if before_analyzer.graph else None
    after_func = after_analyzer.graph.get_main_function() if after_analyzer.graph else None
    
    before_counts = before_analyzer.count_nodes()
    after_counts = after_analyzer.count_nodes()
    
    graph_type_changed = False
    if before_func and after_func:
        graph_type_changed = before_func.graphtype != after_func.graphtype
    
    return {
        'operations_diff': after_counts['operations'] - before_counts['operations'],
        'tensors_diff': after_counts['tensors'] - before_counts['tensors'],
        'rawtensors_diff': after_counts['rawtensors'] - before_counts['rawtensors'],
        'subgraph_count_diff': after_counts['subgraphs'] - before_counts['subgraphs'],
        'before_counts': before_counts,
        'after_counts': after_counts,
        'graph_type_changed': graph_type_changed
    }


def print_tensor_info(tensor: TensorInfo):
    """打印Tensor信息"""
    print(f"\n=== Tensor[{tensor.magic}] ===")
    print(f"Shape: {tensor.shape}")
    print(f"ValidShape: {tensor.validshape}")
    print(f"Offset: {tensor.offset}")
    print(f"RawTensor: {tensor.rawtensor}")
    print(f"NodeType: {tensor.nodetype} ({'Input' if tensor.is_input() else 'Output' if tensor.is_output() else 'Normal'})")
    print(f"MemID: {tensor.mem_id}")
    print(f"MemRange: {tensor.mem_range}")
    mem_asis, mem_tobe = tensor.get_memory_type_str()
    print(f"MemType: {mem_asis} -> {mem_tobe}")
    print(f"SubgraphID: {tensor.subgraphid}")
    print(f"SubgraphBoundary: {tensor.subgraph_boundary}")
    print(f"LifeRange: {tensor.life_range}")


def print_operation_info(op: OperationInfo):
    """打印Operation信息"""
    print(f"\n=== Operation[{op.opmagic}] ===")
    print(f"Opcode: {op.opcode}")
    print(f"Inputs: {op.ioperands}")
    print(f"Outputs: {op.ooperands}")
    print(f"Attr: {op.attr}")
    print(f"SubgraphID: {op.subgraphid}")
    print(f"Latency: {op.latency}")
    print(f"Kind: {op.kind}")
    if op.tile:
        print(f"Tile: {op.tile}")
    if op.op_attr:
        print(f"OpAttr: {op.op_attr}")
    if op.file:
        print(f"Source: {op.file}:{op.line}")


def print_summary(summary: Dict[str, Any]):
    """打印摘要信息"""
    print("\n" + "="*60)
    print("计算图摘要")
    print("="*60)
    print(f"EntryHash: {summary.get('entryhash', 'N/A')}")
    print(f"Version: {summary.get('version', 'N/A')}")
    print(f"Function: {summary.get('function_name', 'N/A')}")
    print(f"FunctionMagic: {summary.get('function_magic', 'N/A')}")
    print(f"GraphType: {summary.get('graph_type', 'N/A')}")
    
    print("\n节点统计:")
    node_counts = summary.get('node_counts', {})
    print(f"  Operations: {node_counts.get('operations', 0)}")
    print(f"  Tensors: {node_counts.get('tensors', 0)}")
    print(f"  RawTensors: {node_counts.get('rawtensors', 0)}")
    print(f"  Subgraphs: {node_counts.get('subgraphs', 0)}")
    
    print("\n输入Tensor:", summary.get('input_tensors', []))
    print("输出Tensor:", summary.get('output_tensors', []))


# 示例使用
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python computation_graph_analyzer.py <json_file_path> [tensor_magic]")
        print("示例:")
        print("  python computation_graph_analyzer.py graph.json")
        print("  python computation_graph_analyzer.py graph.json 28")
        sys.exit(1)
    
    json_path = sys.argv[1]
    
    # 创建分析器
    analyzer = ComputationGraphAnalyzer()
    
    # 加载计算图
    try:
        graph = analyzer.load_graph(json_path)
        print(f"成功加载计算图: {json_path}")
    except Exception as e:
        print(f"加载计算图失败: {e}")
        sys.exit(1)
    
    # 打印摘要
    summary = analyzer.get_summary()
    print_summary(summary)
    
    # 如果指定了Tensor magic，显示详细信息
    if len(sys.argv) > 2:
        tensor_magic = int(sys.argv[2])
        tensor = analyzer.find_tensor_by_magic(tensor_magic)
        
        if tensor:
            print_tensor_info(tensor)
            
            # 查找生产者
            producer = analyzer.find_producer_of_tensor(tensor_magic)
            if producer:
                print("\n生产者:")
                print_operation_info(producer)
            
            # 查找消费者
            consumers = analyzer.find_consumers_of_tensor(tensor_magic)
            if consumers:
                print(f"\n消费者 ({len(consumers)} 个):")
                for consumer in consumers:
                    print_operation_info(consumer)
            
            # 追踪数据流
            flow = analyzer.trace_data_flow(tensor_magic)
            print(f"\n数据流路径: {flow}")
        else:
            print(f"\n未找到 Tensor[{tensor_magic}]")
