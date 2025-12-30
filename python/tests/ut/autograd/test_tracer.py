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
"""
Unit tests for autograd tracer and hooks.

Tests M1 milestone requirements:
- Hook + Tracer can stably produce graph
- Zero behavior change when autograd is not enabled
"""
import pytest


class TestAutogradContext:
    """Test autograd context management."""

    def test_is_tracing_default_false(self):
        """Test that is_tracing returns False by default."""
        from pypto.autograd.context import is_tracing
        assert not is_tracing()

    def test_trace_context_manager(self):
        """Test trace context manager enables tracing."""
        from pypto.autograd.context import is_tracing, trace
        from pypto.autograd.tracer import Tracer

        assert not is_tracing()

        tracer = Tracer()
        with trace(tracer):
            assert is_tracing()

        assert not is_tracing()

    def test_no_trace_context_manager(self):
        """Test no_trace context manager disables tracing."""
        from pypto.autograd.context import is_tracing, trace, no_trace
        from pypto.autograd.tracer import Tracer

        tracer = Tracer()
        with trace(tracer):
            assert is_tracing()

            with no_trace():
                assert not is_tracing()

            assert is_tracing()

    def test_nested_trace_contexts(self):
        """Test nested trace context managers."""
        from pypto.autograd.context import is_tracing, trace, get_current_tracer
        from pypto.autograd.tracer import Tracer

        tracer1 = Tracer(name="tracer1")
        tracer2 = Tracer(name="tracer2")

        with trace(tracer1):
            assert get_current_tracer() is tracer1

            with trace(tracer2):
                assert get_current_tracer() is tracer2

            assert get_current_tracer() is tracer1

        assert get_current_tracer() is None


class TestAutogradGraph:
    """Test autograd graph IR."""

    def test_create_value(self):
        """Test Value creation."""
        from pypto.autograd.graph import Graph

        graph = Graph("test")
        value = graph.create_value(tensor_id=1, shape=[2, 3], dtype="fp32", name="x")

        assert value.id == 0
        assert value.tensor_id == 1
        assert value.version == 0
        assert value.shape == [2, 3]
        assert value.dtype == "fp32"
        assert value.name == "x"

    def test_create_value_with_version(self):
        """Test Value creation with version increment (SSA)."""
        from pypto.autograd.graph import Graph

        graph = Graph("test")
        v0 = graph.create_value(tensor_id=1, name="x_v0")
        v1 = graph.create_value(tensor_id=1, name="x_v1", increment_version=True)

        assert v0.tensor_id == 1
        assert v0.version == 0
        assert v1.tensor_id == 1
        assert v1.version == 1
        assert v0.id != v1.id

    def test_create_node(self):
        """Test Node creation."""
        from pypto.autograd.graph import Graph, NodeType

        graph = Graph("test")
        v_in = graph.create_value(tensor_id=1, name="input")
        v_out = graph.create_value(tensor_id=2, name="output")

        node = graph.create_node(
            node_type=NodeType.OP,
            op_name="add",
            inputs=[v_in],
            outputs=[v_out],
            attrs={"alpha": 1.0},
        )

        assert node.node_type == NodeType.OP
        assert node.op_name == "add"
        assert len(node.inputs) == 1
        assert len(node.outputs) == 1
        assert node.attrs["alpha"] == 1.0
        assert v_out.producer is node
        assert node in v_in.consumers

    def test_reverse_topological_order(self):
        """Test reverse topological order for backward pass."""
        from pypto.autograd.graph import Graph, NodeType

        graph = Graph("test")

        # Create a simple graph: x -> add -> y -> mul -> z
        x = graph.create_value(tensor_id=1, name="x")
        y = graph.create_value(tensor_id=2, name="y")
        z = graph.create_value(tensor_id=3, name="z")

        add_node = graph.create_node(
            node_type=NodeType.OP,
            op_name="add",
            inputs=[x],
            outputs=[y],
        )
        mul_node = graph.create_node(
            node_type=NodeType.OP,
            op_name="mul",
            inputs=[y],
            outputs=[z],
        )

        reverse_order = graph.reverse_topological_order()
        assert len(reverse_order) == 2
        assert reverse_order[0] is mul_node  # mul comes first in reverse order
        assert reverse_order[1] is add_node


class TestAutogradTracer:
    """Test autograd tracer."""

    def test_tracer_init(self):
        """Test Tracer initialization."""
        from pypto.autograd.tracer import Tracer

        tracer = Tracer(name="test_tracer")
        assert tracer.graph.name == "test_tracer"
        assert len(tracer.graph.nodes) == 0

    def test_record_input(self):
        """Test recording graph inputs."""
        from pypto.autograd.tracer import Tracer
        from pypto.autograd.graph import NodeType

        tracer = Tracer()

        # Create a mock tensor-like object
        class MockTensor:
            id = 42
            shape = [2, 3]
            dtype = "fp32"

        mock_tensor = MockTensor()
        value = tracer.record_input(mock_tensor, name="weight")

        assert value.tensor_id == 42
        assert value.name == "weight"
        assert len(tracer.graph.nodes) == 1
        assert tracer.graph.nodes[0].node_type == NodeType.INPUT

    def test_record_op_binds_names_and_attrs(self):
        """Tracer.record_op should bind signature names and capture attrs."""
        from pypto.autograd.tracer import Tracer

        tracer = Tracer()

        class MockTensor:
            def __init__(self, tid: int):
                self._base = object()
                self.id = tid
                self.shape = [2, 3]
                self.dtype = "DT_FP32"

        a = MockTensor(1)
        b = MockTensor(2)
        out = MockTensor(3)

        def add(input, other, *, alpha=1):
            return out

        tracer.record_op(
            op_callable=add,
            op_name="Add",
            args=(a, b),
            kwargs={"alpha": 2.0},
            outputs=out,
        )

        node = tracer.graph.nodes[-1]
        assert node.op_name == "Add"
        assert node.input_names == ["input", "other"]
        assert node.output_names == ["output"]
        assert node.attrs["alpha"] == 2.0
        assert node.saved_tensors["input"] is a
        assert node.saved_tensors["other"] is b
        assert node.saved_tensors["output"] is out

    def test_record_op_assemble_inplace_ssa(self):
        """Assemble should create a new SSA version for the out tensor."""
        from pypto.autograd.tracer import Tracer

        tracer = Tracer()

        class MockTensor:
            def __init__(self, tid: int, shape):
                self._base = object()
                self.id = tid
                self.shape = list(shape)
                self.dtype = "DT_FP32"

        src = MockTensor(1, [2, 2])
        out = MockTensor(2, [4, 4])
        offsets = [0, 0]

        def assemble(input, offsets, out):
            return None

        tracer.record_op(
            op_callable=assemble,
            op_name="Assemble",
            args=(src, offsets, out),
            kwargs={},
            outputs=None,
        )

        node = tracer.graph.nodes[-1]
        assert node.op_name == "Assemble"
        assert node.input_names == ["input", "out"]
        assert node.output_names == ["output"]
        assert len(node.inputs) == 2
        assert len(node.outputs) == 1

        # out tensor should have two versions (v0 as input, v1 as output)
        versions = tracer.graph.get_all_versions(out.id)
        assert len(versions) == 2
        assert versions[0].version == 0
        assert versions[1].version == 1
        assert node.inputs[1] is versions[0]
        assert node.outputs[0] is versions[1]


class TestAutogradDefenses:
    """Test autograd defense mechanisms."""

    def test_control_flow_defense_loop(self):
        """Test that loop raises NotImplementedError in autograd mode."""
        from pypto.autograd.context import trace
        from pypto.autograd.tracer import Tracer

        tracer = Tracer()
        with trace(tracer):
            # Import here to avoid issues if pypto is not fully installed
            try:
                from pypto._controller import _check_autograd_control_flow
                with pytest.raises(NotImplementedError, match="control flow"):
                    _check_autograd_control_flow("loop")
            except ImportError:
                pytest.skip("pypto not fully installed")

    def test_control_flow_defense_cond(self):
        """Test that cond raises NotImplementedError in autograd mode."""
        from pypto.autograd.context import trace
        from pypto.autograd.tracer import Tracer

        tracer = Tracer()
        with trace(tracer):
            try:
                from pypto._controller import _check_autograd_control_flow
                with pytest.raises(NotImplementedError, match="control flow"):
                    _check_autograd_control_flow("cond")
            except ImportError:
                pytest.skip("pypto not fully installed")

    def test_no_defense_when_not_tracing(self):
        """Test that defense checks pass when not in autograd mode."""
        from pypto.autograd.context import is_tracing

        assert not is_tracing()

        try:
            from pypto._controller import _check_autograd_control_flow
            # Should not raise when not tracing
            _check_autograd_control_flow("loop")
            _check_autograd_control_flow("cond")
        except ImportError:
            pytest.skip("pypto not fully installed")


class TestZeroBehaviorChange:
    """Test that autograd hooks have zero behavior change when not tracing."""

    def test_op_wrapper_no_overhead_when_not_tracing(self):
        """Test that op_wrapper has no observable change when not tracing."""
        from pypto.autograd.context import is_tracing

        # Ensure we're not tracing
        assert not is_tracing()

        # Operations should work normally
        # (This is a basic smoke test - full tests require pypto compilation)

    def test_module_lazy_import(self):
        """Test that autograd module uses lazy import."""
        import pypto

        # Access autograd should trigger lazy import
        autograd = pypto.autograd
        assert autograd is not None
        assert hasattr(autograd, "Tracer")
        assert hasattr(autograd, "trace")
        assert hasattr(autograd, "no_trace")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
