#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Autograd IR (Intermediate Representation)."""
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

__all__ = ["Value", "Node", "NodeType", "Graph"]


class NodeType(Enum):
    OP = auto()
    MOVE = auto()
    INPUT = auto()
    CONSTANT = auto()
    CALL = auto()
    IF = auto()
    LOOP = auto()


@dataclass
class Value:
    """Represents a tensor value in the autograd graph (SSA form)."""
    id: int
    tensor_id: int
    version: int = 0
    shape: Optional[List[Any]] = None
    dtype: Optional[Any] = None
    producer: Optional["Node"] = None
    consumers: List["Node"] = field(default_factory=list)
    name: str = ""

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, Value):
            return False
        return self.id == other.id

    def __repr__(self):
        name_str = f" '{self.name}'" if self.name else ""
        return f"Value(id={self.id}, tensor={self.tensor_id}, v{self.version}{name_str})"


@dataclass
class Node:
    """Represents a computation node in the autograd graph."""
    id: int
    node_type: NodeType
    op_name: str = ""
    op_callable: Optional[Callable] = None
    inputs: List[Value] = field(default_factory=list)
    outputs: List[Value] = field(default_factory=list)
    attrs: Dict[str, Any] = field(default_factory=dict)
    source_location: Optional[Tuple[str, int]] = None
    input_names: List[str] = field(default_factory=list)
    output_names: List[str] = field(default_factory=list)
    saved_tensors: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.id == other.id

    def __repr__(self):
        inputs_str = ", ".join(str(v.id) for v in self.inputs)
        outputs_str = ", ".join(str(v.id) for v in self.outputs)
        return f"Node({self.op_name}, inputs=[{inputs_str}], outputs=[{outputs_str}])"


class Graph:
    """The autograd computation graph."""

    def __init__(self, name: str = "autograd_graph"):
        self.name = name
        self._values: Dict[int, Value] = {}
        self._nodes: List[Node] = []
        self._value_counter = 0
        self._node_counter = 0
        self._tensor_to_value: Dict[int, Value] = {}
        self._tensor_versions: Dict[int, List[Value]] = {}

    def create_value(
        self,
        tensor_id: int,
        shape: Optional[List[Any]] = None,
        dtype: Optional[Any] = None,
        name: str = "",
        increment_version: bool = False,
    ) -> Value:
        if tensor_id in self._tensor_versions:
            versions = self._tensor_versions[tensor_id]
            if increment_version:
                version = len(versions)
            else:
                current = self._tensor_to_value.get(tensor_id)
                if current is not None:
                    return current
                version = len(versions)
        else:
            version = 0
            self._tensor_versions[tensor_id] = []

        value = Value(
            id=self._value_counter,
            tensor_id=tensor_id,
            version=version,
            shape=shape,
            dtype=dtype,
            name=name,
        )
        self._value_counter += 1
        self._values[value.id] = value
        self._tensor_versions[tensor_id].append(value)
        self._tensor_to_value[tensor_id] = value
        return value

    def get_value(self, tensor_id: int) -> Optional[Value]:
        return self._tensor_to_value.get(tensor_id)

    def get_all_versions(self, tensor_id: int) -> List[Value]:
        return self._tensor_versions.get(tensor_id, [])

    def create_node(
        self,
        node_type: NodeType,
        op_name: str = "",
        op_callable: Optional[Callable] = None,
        inputs: Optional[List[Value]] = None,
        outputs: Optional[List[Value]] = None,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        saved_tensors: Optional[Dict[str, Any]] = None,
        attrs: Optional[Dict[str, Any]] = None,
        source_location: Optional[Tuple[str, int]] = None,
    ) -> Node:
        node = Node(
            id=self._node_counter,
            node_type=node_type,
            op_name=op_name,
            op_callable=op_callable,
            inputs=inputs or [],
            outputs=outputs or [],
            input_names=input_names or [],
            output_names=output_names or [],
            saved_tensors=saved_tensors or {},
            attrs=attrs or {},
            source_location=source_location,
        )
        self._node_counter += 1
        self._nodes.append(node)

        for output in node.outputs:
            output.producer = node
        for input_val in node.inputs:
            input_val.consumers.append(node)

        return node

    @property
    def nodes(self) -> List[Node]:
        return self._nodes

    @property
    def values(self) -> Dict[int, Value]:
        return self._values

    def get_inputs(self) -> List[Value]:
        return [v for v in self._values.values() if v.producer is None]

    def get_outputs(self) -> List[Value]:
        return [v for v in self._values.values() if not v.consumers]

    def reverse_topological_order(self) -> List[Node]:
        return list(reversed(self._nodes))

    def __repr__(self):
        return f"Graph(name={self.name}, nodes={len(self._nodes)}, values={len(self._values)})"
