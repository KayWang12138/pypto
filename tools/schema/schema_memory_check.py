from enum import Enum
from typing import List, Dict, Union
from typing import Optional, Callable
from collections import OrderedDict
import copy
import os
import argparse

DEV_TRACE_PREFIX = "#trace:"
SCHEMA_ADDRESS_PREFIX = "0x"
INVALID_TRACE_TASK_DEPEND_INDEX = -1
TASKID_TASK_BITS = 20
TASKID_TASK_MASK = (1 << TASKID_TASK_BITS) - 1

class TraceMemoryRange:
    def __init__(self, begin: int = 0, end: int = 0):
        self._begin = begin
        self._end = end

    @property
    def begin(self) -> int:
        return self._begin

    @property
    def end(self) -> int:
        return self._end

    def __eq__(self, rhs: 'TraceMemoryRange') -> bool:
        if not isinstance(rhs, TraceMemoryRange):
            return False
        return self.begin == rhs.begin and self.end == rhs.end

class TraceRawTensorMemory:
    def __init__(self, memory_range: TraceMemoryRange = None, shape: list[int] = None):
        if memory_range is None:
            self._memory_range = TraceMemoryRange()
        else:
            self._memory_range = memory_range
        if shape is None:
            self._shape = []
        else:
            self._shape = shape
    
    @property
    def memory_range(self) -> TraceMemoryRange:
        return self._memory_range
    
    @memory_range.setter
    def memory_range(self, memory_range):
        self._memory_range = memory_range
    
    @property
    def shape(self) -> list[int]:
        return self._shape

    @shape.setter
    def shape(self, shape: list[int]):
        self._shape = shape

class TraceCopy:
    def __init__(
        self,
        is_copy_out: bool,
        raw_tensor: Optional[TraceRawTensorMemory],
        offset: List[int],
        shape: List[int],
        is_atomic_add: bool = False
    ):
        self._is_copy_out = is_copy_out
        self._raw_tensor = raw_tensor
        self._offset = offset.copy()
        self._shape = shape.copy()
        self._is_atomic_add = is_atomic_add

    @property
    def is_copy_out(self) -> bool:
        return self._is_copy_out

    @property
    def raw_tensor(self) -> Optional[TraceRawTensorMemory]:
        return self._raw_tensor
    
    @property
    def offset(self):
        return self._offset.copy()
    
    @property
    def shape(self) -> List[int]:
        return self._shape.copy()
    
    @property
    def is_atomic_add(self):
        return self._is_atomic_add
    
    @offset.setter
    def offset(self, offset: List[int]):
        self._offset = offset.copy()
    
    @shape.setter
    def shape(self, shape: List[int]):
        self._shape = shape.copy()
    
    @is_atomic_add.setter
    def is_atomic_add(self, is_atomic_add: bool):
        self._is_atomic_add = is_atomic_add
    
    def overlap(src: 'TraceCopy', dst: 'TraceCopy') -> bool:
        src_range = src.raw_tensor.memory_range
        dst_range = dst.raw_tensor.memory_range

        if src_range.end <= dst_range.begin:
            return False
        if dst_range.end <= src_range.begin:
            return False
        assert src_range == dst_range, "memory reuse must happen for full match."
        assert len(src.offset) == len(dst.offset), "memory reuse must happen for same dimension."

        for dim in range(len(src.offset)):
            src_offset = src.offset[dim]
            src_shape = src.shape[dim]
            dst_offset = dst.offset[dim]
            dst_shape = dst.shape[dim]

            if src_offset + src_shape <= dst_offset:
                return False
            if dst_offset + dst_shape <= src_offset:
                return False
        return True
    
class TraceLeafTaskUid:
    def __init__(self, device_task_index=-1, dup_index=-1, root_index=-1, 
                 operation_index=-1, leaf_index=-1):
        self._device_task_index = device_task_index
        self._dup_index = dup_index
        self._root_index = root_index
        self._operation_index = operation_index
        self._leaf_index = leaf_index

    @property
    def device_task_index(self):
        return self._device_task_index

    @property
    def dup_index(self):
        return self._dup_index

    @property
    def root_index(self):
        return self._root_index

    @property
    def operation_index(self):
        return self._operation_index

    @property
    def leaf_index(self):
        return self._leaf_index

    def get_task_id(self):
        return self._dup_index << TASKID_TASK_BITS | self._operation_index
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, TraceLeafTaskUid):
            return False
        return (self._device_task_index == other._device_task_index and
                self._dup_index == other._dup_index and
                self._root_index == other._root_index and
                self._operation_index == other._operation_index and
                self._leaf_index == other._leaf_index)
    
    def __hash__(self) -> int:
        return hash((self._device_task_index,
                     self._dup_index,
                     self._root_index,
                     self._operation_index,
                     self._leaf_index))

class TraceRootTaskUid:
    def __init__(self, device_task_index=-1, dup_index=-1, root_index=-1):
        self._device_task_index = device_task_index
        self._dup_index = dup_index
        self._root_index = root_index

    @property
    def device_task_index(self):
        return self._device_task_index

    @property
    def dup_index(self):
        return self._dup_index

    @property
    def root_index(self):
        return self._root_index
    
    def __hash__(self):
        return hash((self._device_task_index, self._dup_index, self._root_index))

    def __eq__(self, other):
        if not isinstance(other, TraceRootTaskUid):
            return False  
        return (self._device_task_index == other._device_task_index and
                self._dup_index == other._dup_index and
                self._root_index == other._root_index)

class TraceDeviceTaskUid:
    def __init__(self, device_task_index: int = -1):
        self._device_task_index = device_task_index
    
    @property
    def device_task_index(self) -> int:
        return self._device_task_index

    def __eq__(self, other: 'TraceDeviceTaskUid') -> bool:
        if not isinstance(other, TraceDeviceTaskUid):
            return False
        return self._device_task_index == other._device_task_index

    def __hash__(self) -> int:
        return hash(self._device_task_index)
    
class TraceCoa:
    def __init__(self, value: int, is_expr: bool = False):
        self._value = value
        self._is_expr = is_expr

    @property
    def value(self) -> int:
        return self._value

    @property
    def is_expr(self) -> bool:
        return self._is_expr

    def __eq__(self, other: 'TraceCoa') -> bool:
        if not isinstance(other, TraceCoa):
            return False
        return self._is_expr == other._is_expr and self._value == other._value

class TraceLeafTask:
    def __init__(self, uid: TraceLeafTaskUid = None):
        self._uid = uid if uid is not None else TraceLeafTaskUid()
        self._coa_list = []
        self._copy_in_list = []
        self._copy_out_list = []
        self._pred_set = set() 
        self._succ_set = set() 

    @property
    def uid(self) -> TraceLeafTaskUid:
        return self._uid
    
    @uid.setter
    def uid(self, value):
        self._uid = value

    @property
    def coa_list(self) -> list:
        return self._coa_list

    @coa_list.setter
    def coa_list(self, value):
        self._coa_list = value

    @property
    def copy_in_list(self) -> list:
        return self._copy_in_list

    @property
    def copy_out_list(self) -> list:
        return self._copy_out_list

    @copy_out_list.setter
    def copy_out_list(self, value):
        self._copy_out_list = copy.deepcopy(value)

    @property
    def pred_set(self) -> set:
        return self._pred_set

    @property
    def succ_set(self) -> set:
        return self._succ_set

    def add_pred(self, pred: TraceLeafTaskUid) -> None:
        self._pred_set.add(pred)

    def add_succ(self, succ: TraceLeafTaskUid) -> None:
        self._succ_set.add(succ)

class TraceRootTaskRawTensorDesc:
    def __init__(self, location: int = -1, offset_or_index: int = 0, size: int = 0):
        self._location = location  
        self._offset_or_index = offset_or_index  
        self._size = size 

    @property
    def location(self) -> int:
        return self._location

    @property
    def offset_or_index(self) -> int:
        return self._offset_or_index

    @property
    def size(self) -> int:
        return self._size

class TraceRootTask:
    def __init__(self, uid=None):
        self._uid = uid if uid is not None else TraceRootTaskUid()
        self._tile_func = None
        self._expr_list = []    
        self._leaf_task_dict = {}  
        self._incast_list = []     
        self._outcast_list = []    
        self._raw_tensor_desc_list = []  
        self._workspace_memory_range = TraceMemoryRange()  

    @property
    def uid(self):
        return self._uid

    @property
    def tile_func(self):
        return self._tile_func

    @tile_func.setter
    def tile_func(self, func):
        self._tile_func = func

    @property
    def expr_list(self):
        return self._expr_list

    @property
    def leaf_task_dict(self):
        return self._leaf_task_dict

    @leaf_task_dict.setter
    def leaf_task_dict(self, value):
        self._leaf_task_dict = value

    @property
    def incast_list(self):
        return self._incast_list  

    @incast_list.setter
    def incast_list(self, value):
        self._incast_list = value  

    @property
    def outcast_list(self):
        return self._outcast_list

    @outcast_list.setter
    def outcast_list(self, value):
        self._outcast_list = value  

    @property
    def raw_tensor_desc_list(self):
        return self._raw_tensor_desc_list

    @property
    def workspace_memory_range(self):
        return self._workspace_memory_range

    @workspace_memory_range.setter
    def workspace_memory_range(self, value):
        self._workspace_memory_range = value

class TraceDependGraph:
    def __init__(self, leaf_task_list=None, leaf_task_depend_index_dict=None, reach_dict=None):
        self._leaf_task_list = leaf_task_list if leaf_task_list is not None else []
        self._leaf_task_depend_index_dict = leaf_task_depend_index_dict if leaf_task_depend_index_dict is not None else {}
        self._reach_dict = reach_dict if reach_dict is not None else []

    @property
    def leaf_task_size(self):
        return len(self._leaf_task_list)

    @property
    def leaf_task_list(self):
        return self._leaf_task_list

    @property
    def leaf_task_depend_index_dict(self):
        return self._leaf_task_depend_index_dict

    @property
    def reach_dict(self):
        return self._reach_dict

    @reach_dict.setter
    def reach_dict(self, value):
        self._reach_dict = value

    def reach(self, src, dst):
        try:
            return self._reach_dict[src][dst] != INVALID_TRACE_TASK_DEPEND_INDEX
        except IndexError:
            return False

class TraceRaceKind(Enum):
    RACE_READ_WRITE = 1
    RACE_WRITE_WRITE = 2
    RACE_ATOMIC_ADD = 3

class TraceRacePart:
    def __init__(self, leaf_task, is_copy_out, copy_index):
        self._leaf_task = leaf_task
        self._is_copy_out = is_copy_out
        self._copy_index = copy_index

    @property
    def leaf_task(self):
        return self._leaf_task

    @property
    def is_copy_out(self):
        return self._is_copy_out

    @property
    def copy_index(self):
        return self._copy_index

class TraceRace:
    def __init__(self, kind, src, dst):
        self._kind = kind
        self._src = src
        self._dst = dst

    @property
    def kind(self):
        return self._kind

    @property
    def src(self):
        return self._src

    @property
    def dst(self):
        return self._dst

class TraceDeviceTask:
    def __init__(self, uid=None):
        self._uid = uid if uid is not None else TraceDeviceTaskUid()
        self._root_task_dict = {}

    @property
    def uid(self):
        return self._uid

    @property
    def root_task_dict(self):
        return self._root_task_dict

    @root_task_dict.setter
    def root_task_dict(self, value):
        self._root_task_dict = value

    def _build_reach_dict(self, graph, depend_index, visit_dict):
        if visit_dict[depend_index]:
            return

        leaf_task = graph.leaf_task_list[depend_index]
        depend_index_dict = graph.leaf_task_depend_index_dict
        reach_dict = graph.reach_dict 

        for succ_uid in leaf_task.succ_set:
            assert succ_uid in depend_index_dict, f"succ_uid {succ_uid} not found in depend_index_dict"
            succ_depend_index = depend_index_dict[succ_uid]
            self._build_reach_dict(graph, succ_depend_index, visit_dict)
            leaf_task_size = graph.leaf_task_size
            for i in range(leaf_task_size):
                if reach_dict[succ_depend_index][i] != INVALID_TRACE_TASK_DEPEND_INDEX:
                    reach_dict[depend_index][i] = succ_depend_index
            reach_dict[depend_index][succ_depend_index] = succ_depend_index
        visit_dict[depend_index] = True

    def build_depend_graph(self):
        leaf_task_list = []
        leaf_task_depend_index_dict = {}
        for ruid, root_task in self._root_task_dict.items():
            for luid, leaf_task in root_task.leaf_task_dict.items():
                leaf_task_depend_index_dict[leaf_task.uid] = len(leaf_task_list)
                leaf_task_list.append(leaf_task)
        leaf_task_size = len(leaf_task_list)
        reach_dict = [
            [INVALID_TRACE_TASK_DEPEND_INDEX for _ in range(leaf_task_size)]
            for _ in range(leaf_task_size)
        ]
        graph = TraceDependGraph(leaf_task_list, leaf_task_depend_index_dict, reach_dict)
        visit_dict = [False] * leaf_task_size
        for i in range(graph.leaf_task_size):
            self._build_reach_dict(graph, i, visit_dict)
        return graph

    def check_race(self, graph):
        race_list = [] 
        leaf_task_size = graph.leaf_task_size
        for src in range(leaf_task_size):
            for dst in range(src + 1, leaf_task_size):
                if graph.reach(src, dst) or graph.reach(dst, src): 
                    continue
                
                src_leaf_task = graph.leaf_task_list[src] 
                dst_leaf_task = graph.leaf_task_list[dst]
                
                src_copy_in_list = src_leaf_task.copy_in_list  
                dst_copy_out_list = dst_leaf_task.copy_out_list
                
                for i, src_copy in enumerate(src_copy_in_list):
                    for j, dst_copy in enumerate(dst_copy_out_list):
                        if TraceCopy.overlap(src_copy, dst_copy):
                            race_part_src = TraceRacePart(src_leaf_task, False, i)
                            race_part_dst = TraceRacePart(dst_leaf_task, True, j)
                            race = TraceRace(TraceRaceKind.RACE_READ_WRITE, race_part_src, race_part_dst)
                            race_list.append(race)
                
                src_copy_out_list = src_leaf_task.copy_out_list
                dst_copy_in_list = dst_leaf_task.copy_in_list
                
                for i, src_copy in enumerate(src_copy_out_list):
                    for j, dst_copy in enumerate(dst_copy_in_list):
                        if TraceCopy.overlap(src_copy, dst_copy):
                            race_part_src = TraceRacePart(src_leaf_task, True, i)
                            race_part_dst = TraceRacePart(dst_leaf_task, False, j)
                            race = TraceRace(TraceRaceKind.RACE_READ_WRITE, race_part_src, race_part_dst)
                            race_list.append(race)
                
                for i, src_copy in enumerate(src_copy_out_list):
                    for j, dst_copy in enumerate(dst_copy_out_list):
                        if TraceCopy.overlap(src_copy, dst_copy):
                            if src_copy.is_atomic_add and dst_copy.is_atomic_add:
                                race_kind = TraceRaceKind.RACE_ATOMIC_ADD
                            else:
                                race_kind = TraceRaceKind.RACE_WRITE_WRITE
                            race_part_src = TraceRacePart(src_leaf_task, True, i)
                            race_part_dst = TraceRacePart(dst_leaf_task, True, j)
                            race = TraceRace(race_kind, race_part_src, race_part_dst)
                            race_list.append(race)
        return race_list

class SchemaNode(list):
    def __init__(self, name: str):
        super().__init__()
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, new_name: str):
        self._name = new_name
    
    @staticmethod
    def parse_schema(schema_input: Union[str, List[str]]) -> List['SchemaNode']:
        if isinstance(schema_input, str):
            schema = schema_input
            pos = schema.find(DEV_TRACE_PREFIX)
            if pos == -1:
                return []
            pos += len(DEV_TRACE_PREFIX)            
            node_list = Parser(schema, pos).parse()
            return node_list
        elif isinstance(schema_input, list):
            schema_list = schema_input
            node_list = []
            for schema in schema_list:
                child_list = SchemaNode.parse_schema(schema)  
                node_list.extend(child_list) 
            return node_list
        else:
            raise TypeError(f"schema_input must be str or list of str, got {type(schema_input)}")

    @staticmethod
    def build_dict(node_list: List['SchemaNode']) -> Dict[str, List['SchemaNode']]:
        node_dict = {}

        def build_schema_dict(dict_ref: Dict[str, List['SchemaNode']], node: 'SchemaNode'):
            node_name = node.name 
            if node_name not in dict_ref:
                dict_ref[node_name] = []
            dict_ref[node_name].append(node)
            for child in node:
                build_schema_dict(dict_ref, child)

        for node in node_list:
            build_schema_dict(node_dict, node)
        return node_dict
    
    def at(self, index: int) -> 'SchemaNode':
        return self[index]

def load_trace_list(node: SchemaNode) -> List[int]:
    res_list = []
    for elt in node:
        name = elt.name
        try:
            num = int(name)
            res_list.append(num)
        except ValueError as e:
            raise ValueError("Node name cannot be converted to an integer") from e

    return res_list

def load_trace_memory_range(node: SchemaNode) -> TraceMemoryRange:
    begin_str = node.at(0).name
    end_str = node.at(1).name
    assert begin_str.startswith(SCHEMA_ADDRESS_PREFIX), \
        f"Invalid starting address prefix: must start with '{SCHEMA_ADDRESS_PREFIX}'"
    assert end_str.startswith(SCHEMA_ADDRESS_PREFIX), \
        f"Invalid prefix for end address: must start with '{SCHEMA_ADDRESS_PREFIX}'"

    try:
        begin = int(begin_str, 16)
        end = int(end_str, 16)
    except ValueError as e:
        raise ValueError(
            f"Address conversion failed: start address / end address is not a valid hexadecimal number"
        ) from e
    return TraceMemoryRange(begin, end)

def load_trace_int(node: SchemaNode) -> int:
    name = node.name
    try:
        value = int(name, 16)
        return value
    except ValueError as e:
        raise ValueError("Node name cannot be converted to an integer") from e

def load_trace_raw_tensor(node: SchemaNode) -> int:
    name = node.name
    assert len(name) > 0 and name[0] == '@', \
        f"Invalid format for node name: must start with '@'"
    try:
        value = int(name[1:], 16)
        return value
    except ValueError as e:
        raise ValueError(f"The part after '@' in node name cannot be converted to an integer") from e

def load_trace_coa_list(node) -> list[TraceCoa]:
    coa_list = []
    for elt in node:
        name = elt.name
        if not name:
            raise ValueError("SchemaNode child node name is empty, failed to parse TraceCoa")
        if name.startswith('?'):
            try:
                value = int(name[1:], 10) 
            except ValueError as e:
                raise ValueError(f"The part after '?' in child node name cannot be converted to an integer") from e
            coa_list.append(TraceCoa(value, is_expr=True))
        else:
            try:
                value = int(name, 10)
            except ValueError as e:
                raise ValueError(f"Child node name cannot be converted to an integer") from e
            coa_list.append(TraceCoa(value))
    return coa_list

def load_trace_succ_list(node) -> list[int]:
    name = node.name
    succ_list = []
    for succ_node in node:
        name = succ_node.name
        if not name:
            raise ValueError("SchemaNode child node name is empty, failed to parse Succ")
        else:
            try:
                value = int(name[1:], 10)
            except ValueError as e:
                raise ValueError(f"Child node name cannot be converted to an integer") from e
            succ_list.append(value)
    return succ_list

rtask_loader_dict: Dict[str, Callable[[TraceRootTask, SchemaNode], None]] = {
    "RActWorkspace": lambda rtask, workspace_node: setattr(
        rtask, 'workspace_memory_range', load_trace_memory_range(workspace_node.at(0))
    ),

    "RActIncastCount": lambda rtask, count_node: [
        rtask.incast_list.clear(),
        rtask.incast_list.extend([TraceRawTensorMemory() for _ in range(load_trace_int(count_node.at(0)))])
    ],

    "RActIncast": lambda rtask, incast_node: (
        setattr(
            rtask.incast_list[load_trace_int(incast_node.at(0).at(0))],
            'memory_range',
            load_trace_memory_range(incast_node.at(1))
        )
    ),

    "RActOutcastCount": lambda rtask, count_node: [
        rtask.outcast_list.clear(),
        rtask.outcast_list.extend([TraceRawTensorMemory() for _ in range(load_trace_int(count_node.at(0)))])
    ],

    "RActOutcast": lambda rtask, outcast_node: (
        setattr(
            rtask.outcast_list[load_trace_int(outcast_node.at(0).at(0))],
            'memory_range',
            load_trace_memory_range(outcast_node.at(1))
        )
    ),

    "RActRawTensorCount": lambda rtask, count_node: (
        rtask.raw_tensor_desc_list.__setitem__(
            slice(None),
            [None for _ in range(load_trace_int(count_node.at(0)))]
        )
    ),

    "RActRawTensor": lambda rtask, desc_node: (
        rtask.raw_tensor_desc_list.__setitem__(
            load_trace_raw_tensor(desc_node.at(0)),
            TraceRootTaskRawTensorDesc(
                location=load_trace_int(desc_node.at(1).at(0)),
                offset_or_index=load_trace_int(desc_node.at(1).at(1)),
                size=load_trace_int(desc_node.at(1).at(2))
            )
        )
    ),
}

ltask_loader_dict: Dict[str, Callable[[TraceLeafTask, SchemaNode], None]] = {
    "LActIncast": lambda ltask, incast_node: (
        ltask._copy_in_list.append(
            TraceCopy(
                is_copy_out=False,
                raw_tensor=TraceRawTensorMemory(memory_range=load_trace_memory_range(incast_node.at(2))),
                offset=load_trace_list(incast_node.at(1).at(0)),
                shape=load_trace_list(incast_node.at(0).at(0)),
                is_atomic_add=False
            )
        )
    ),

    "LActOutcast": lambda ltask, outcast_node: (
        ltask._copy_out_list.append(
            TraceCopy(
                is_copy_out=True,
                raw_tensor=TraceRawTensorMemory(memory_range=load_trace_memory_range(outcast_node.at(2))),
                offset=load_trace_list(outcast_node.at(1).at(0)),
                shape=load_trace_list(outcast_node.at(0).at(0)),
                is_atomic_add=False
            )
        )
    ),
}

class TraceExecution:
    def __init__(self):
        self._leaf_task_dict = {}    
        self._root_task_dict = {}    
        self._device_task_dict = {}  
        self._workspace_spill_range = TraceMemoryRange()

        self._dev_root_list = OrderedDict()
        self._dev_leaf_list = OrderedDict()

    @property
    def leaf_task_dict(self):
        return self._leaf_task_dict

    @property
    def root_task_dict(self):
        return self._root_task_dict

    @property
    def device_task_dict(self):
        return self._device_task_dict

    @property
    def workspace_spill_range(self):
        return self._workspace_spill_range

    def get_leaf_task(self, luid):
        if luid in self._leaf_task_dict:
            return self._leaf_task_dict[luid]

        ltask = TraceLeafTask(luid)
        self._leaf_task_dict[luid] = ltask
        ruid = TraceRootTaskUid(
            device_task_index=luid.device_task_index,
            dup_index=luid.dup_index,
            root_index=luid.root_index
        )
        rtask = self.get_root_task(ruid)
        rtask.leaf_task_dict[luid] = ltask
        return ltask

    def get_root_task(self, ruid):
        if ruid in self._root_task_dict:
            return self._root_task_dict[ruid]

        rtask = TraceRootTask(ruid)
        self._root_task_dict[ruid] = rtask
        duid = TraceDeviceTaskUid(
            ruid.device_task_index
        )
        dtask = self.get_device_task(duid)
        dtask.root_task_dict[ruid] = rtask
        return rtask

    def get_device_task(self, duid):
        if duid in self._device_task_dict:
            return self._device_task_dict[duid]
        dtask = TraceDeviceTask(duid)
        self._device_task_dict[duid] = dtask
        return dtask

    def init_root_list(self, dev_root_list):
        self._dev_root_list = OrderedDict.fromkeys(dev_root_list)

    def init_leaf_list(self, dev_leaf_list):
        self._dev_leaf_list = OrderedDict.fromkeys(dev_leaf_list)
    
    def build_task_successor_dict(self, file_path):
        valid_uid_list = list(self._leaf_task_dict.keys())
        uid_map = {}
        succ_map = {}
        task_info = []
        valid_uid_by_seq = {}
        for uid in valid_uid_list:
            seq_no = uid._device_task_index
            if seq_no not in valid_uid_by_seq:
                valid_uid_by_seq[seq_no] = set()
            valid_uid_by_seq[seq_no].add(uid)

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f if l.strip()][1:]
            for line in lines:
                parts = line.split(',')
                while len(parts) < 10: parts.append("")
                
                seqNo = int(parts[0])
                raw_taskid = int(parts[1])
                root_index = int(parts[2])
                leaf_index = int(parts[5])
                
                dup_index = raw_taskid >> TASKID_TASK_BITS
                op_index = raw_taskid & TASKID_TASK_MASK
                
                uid_str = f"{seqNo}_{dup_index}_{root_index}_{op_index}_{leaf_index}"
                task_uid = TraceLeafTaskUid(seqNo, dup_index, root_index, op_index, leaf_index)
                uid_map[uid_str] = task_uid
                task_info.append((seqNo, raw_taskid, uid_str, task_uid))
                
                succ_list = []
                for s in ','.join(parts[9:]).split(','):
                    s_strip = s.strip()
                    if s_strip:
                        succ_list.append(int(s_strip))
                succ_map[uid_str] = succ_list

        def get_real_succ(uid_str, visited):
            if uid_str in visited or uid_str not in succ_map:
                return []
            visited.add(uid_str)
            real_succ = []
            
            current_seq = None
            current_uid = None
            for (s_no, _, u_str, task_uid) in task_info:
                if u_str == uid_str:
                    current_seq = s_no
                    current_uid = task_uid
                    break
            if current_seq is None or current_uid is None:
                return []

            for succ_tid in succ_map[uid_str]:
                for (s_no, t_id, u_str, task_uid) in task_info:
                    if t_id == succ_tid and u_str not in visited and s_no == current_seq :
                        if task_uid in valid_uid_by_seq[s_no]:
                            real_succ.append(task_uid)
                        else:
                            real_succ.extend(get_real_succ(u_str, visited))
            return list({uid: None for uid in real_succ}.keys())

        result = {}
        valid_uid_str = {
            f"{uid._device_task_index}_{uid._dup_index}_{uid._root_index}_{uid._operation_index}_{uid._leaf_index}": uid
            for uid in valid_uid_list
        }
        
        for uid_str, uid in valid_uid_str.items():
            result[uid] = get_real_succ(uid_str, set())

        return result

    def load_trace(self, trace):
        trace_node_list = SchemaNode.parse_schema(trace)
        for trace_node in trace_node_list:
            node_dict = SchemaNode.build_dict([trace_node])
            if "DEvent" in node_dict:
                pass
            elif "REvent" in node_dict:
                ruid_node = node_dict["RUid"][0]
                device_task_index = int(ruid_node.at(0x0).name)
                dup_index = int(ruid_node.at(0x1).name)
                root_index = int(ruid_node.at(0x2).name)
                ruid = TraceRootTaskUid(device_task_index, dup_index, root_index)
                rtask = self.get_root_task(ruid)
                for key, loader in rtask_loader_dict.items():
                    if key in node_dict:
                        node = node_dict[key][0]
                        loader(rtask, node)
            elif "LEvent" in node_dict:
                luid_node = node_dict["LUid"][0]
                device_task_index = int(luid_node.at(0x0).name)
                dup_index = int(luid_node.at(0x1).name)
                root_index = int(luid_node.at(0x2).name)
                operation_index = int(luid_node.at(0x3).name)
                leaf_index = int(luid_node.at(0x4).name)
                luid = TraceLeafTaskUid(
                    device_task_index, dup_index, root_index,
                    operation_index, leaf_index
                )
                ltask = self.get_leaf_task(luid)
                for key, loader in ltask_loader_dict.items():
                    if key in node_dict:
                        node = node_dict[key][0]
                        loader(ltask, node) 

class Parser:
    def __init__(self, text: str, base: int = 0):
        self.text = text
        self.base = base
        self.token_list = []
        self.pos = 0

    class Token:
        ID = 0
        def __init__(self, kind: int, text: str = ""):
            self.kind = kind
            self.text = text

        def Kind(self) -> int:
            return self.kind

        def Text(self) -> str:
            return self.text

        def __repr__(self):
            return f"Token(kind={self.kind}, text='{self.text}')"

    def tokenization(self):
        self.token_list.clear()
        curr = ""  
        for idx in range(self.base, len(self.text)):
            char = self.text[idx]
            if char in ('#', ',', '[', ']', '{', '}'):
                if curr:
                    self.token_list.append(self.Token(self.Token.ID, curr))
                    curr = ""
                self.token_list.append(self.Token(ord(char), char))

            elif char == ' ':
                if curr:
                    self.token_list.append(self.Token(self.Token.ID, curr))
                    curr = ""
            else:
                curr += char
        if curr:
            self.token_list.append(self.Token(self.Token.ID, curr))

    def current(self) -> Token:
        return self.token_list[self.pos]

    def accessible(self) -> bool:
        return self.pos < len(self.token_list)

    def move_next(self):
        self.pos += 1

    def parse_node(self) -> SchemaNode:
        curr_node: Optional[SchemaNode] = None
        current_token = self.current()
        if current_token.Kind() == ord('['):
            curr_node = SchemaNode("")
            self.move_next()

            if self.current().Kind() == ord(']'):
                self.move_next()
                return curr_node
            
            while True:
                child_node = self.parse_node()
                curr_node.append(child_node)
                if self.current().Kind() == ord(','):
                    self.move_next()  
                elif self.current().Kind() == ord(']'):
                    self.move_next()  
                    break
                else:
                    assert False, f"Invalid format at pos {self.pos}: expected , or ], got {self.current()}"
        else:
            assert current_token.Kind() == self.Token.ID, f"Expected ID token at pos {self.pos}, got {current_token}"
            curr_node = SchemaNode(current_token.Text())
            self.move_next()
            if self.current().Kind() == ord('{'):
                self.move_next()

                if self.current().Kind() == ord('}'):
                    self.move_next()
                    return curr_node
                
                while True:
                    child_node = self.parse_node()
                    curr_node.append(child_node)
                    if self.current().Kind() == ord(','):
                        self.move_next()  
                    elif self.current().Kind() == ord('}'):
                        self.move_next()
                        break
                    else:
                        assert False, f"Invalid format at pos {self.pos}: expected , or }}, got {self.current()}"

            elif self.current().Kind() in (ord(','), ord(']'), ord('}')):
                pass
            else:
                assert False, f"Invalid format after ID at pos {self.pos}: got {self.current()}"
        return curr_node

    def parse(self) -> List[SchemaNode]:
        self.tokenization()
        self.pos = 0
        node_list = []
        while self.accessible():
            while self.accessible() and self.current().Kind() != ord('#'):
                self.move_next()
            if not self.accessible():
                break
            assert self.current().Kind() == ord('#'), f"Expected # at pos {self.pos}, got {self.current()}"
            self.move_next()
            node_list.append(self.parse_node())
        return node_list

class LoadTraceLog:
    def __init__(self, path: str):
        self.path = path
        self.trace_content_list: List[str] = []

    def read_single_file(self, file_path: str, encoding: str, strip_whitespace: bool) -> None:
        try:
            with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    trace_pos = line.find('#trace')
                    if trace_pos != -1:
                        trace_content = line[trace_pos:]
                        trace_content = trace_content.replace('"', '')
                        trace_content = trace_content.replace('#trace:  ', '#trace:')
                        if strip_whitespace:
                            trace_content = trace_content.strip()
                        self.trace_content_list.append(trace_content)
        except Exception as e:
            raise RuntimeError("Failed to read file") from e

    def read_trace_lines(self, encoding: str = 'utf-8', strip_whitespace: bool = True) -> List[str]:
        self.trace_content_list.clear()
        if os.path.isfile(self.path):
            self.read_single_file(self.path, encoding, strip_whitespace)
        elif os.path.isdir(self.path):
            for file_name in os.listdir(self.path):
                file_path = os.path.join(self.path, file_name)
                if os.path.isfile(file_path):
                    self.read_single_file(file_path, encoding, strip_whitespace)

        return self.trace_content_list

    def get_trace_count(self) -> int:
        return len(self.trace_content_list)

def CheckLeafRace(device_log_path: str, topo_file_path: str):
    log_loador = LoadTraceLog(device_log_path)
    log_lines = log_loador.read_trace_lines()
    exec = TraceExecution()
    for line in log_lines:
        if 'coa' not in line and 'expr' not in line:
            exec.load_trace(line)
    
    res = exec.build_task_successor_dict(topo_file_path)

    for key, value in res.items():
        for item in value:
            exec.get_leaf_task(key).add_succ(item)

    for key, value in exec.leaf_task_dict.items():
        rtask_uid = TraceRootTaskUid(key.device_task_index, key.dup_index, key.root_index)
        root_task = exec.get_root_task(rtask_uid)
        root_task.leaf_task_dict[key] = value

    for key, value in exec.root_task_dict.items():
        dtask_uid = TraceDeviceTaskUid(key.device_task_index)
        device_task = exec.get_device_task(dtask_uid)
        device_task.root_task_dict[key] = value
    
    for key, value in exec.device_task_dict.items():
        degGraph = value.build_depend_graph()
        race_list = value.check_race(degGraph)
        
        if len(race_list) > 0:
            for race in race_list:
                src_task_id = race.src.leaf_task.uid.get_task_id()
                dst_task_id = race.dst.leaf_task.uid.get_task_id()
                print(f"[Error] DeviceTask: {key.device_task_index}, race kind: {race.kind}, src leaf task: {src_task_id}, dst leaf task: {dst_task_id}")
        else:
            print(f"[Success] Memory overlap was not detected in leaf func of device task {key.device_task_index}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Schema trace log analysis",
        formatter_class=argparse.RawTextHelpFormatter 
    )
    parser.add_argument(
        "-d", "--device-log", 
        dest="device_log_path",
        required=True,
        help="Path to device log"
    )
    parser.add_argument(
        "-t", "--topo-file",
        dest="topo_file_path",
        required=True,
        help="Path to topology file"
    )
    args = parser.parse_args()
    CheckLeafRace(args.device_log_path, args.topo_file_path)