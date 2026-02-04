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
"""

class JitCompiledFunction:
    """Holds a compiled function."""

    export_provider: ExportProvider = None

    def __init__(
        self,
        ir_module,
        engine,
        capi_func,
        args_spec,
        function_name,
        kernel_info,
        jit_time_profiling,
        jit_function_artifacts,
        prefix=None,
        load_from_binary=False,
        dynamic_args=None,
        dynamic_kwargs=None,
    ):
        self.ir_module = ir_module
        self.engine = engine
        self.capi_func = capi_func
        self.function_name = function_name
        self.kernel_info = kernel_info
        if args_spec is not None:
            self.args_spec = ExecutionArgs(args_spec, self.function_name)
        self.jit_time_profiling = jit_time_profiling

        assert (
            isinstance(jit_function_artifacts, JitFunctionArtifacts)
            or jit_function_artifacts is None
        )
        self.artifacts = jit_function_artifacts
        self.prefix = prefix
        self.load_from_binary = load_from_binary

        # This runtime state is stored here so that we can preserve the module
        # in the compiler cache. Callers can extend the lifetime of the module
        # by creating and retaining the executor.
        self.jit_module = None
        self._executor_lock = threading.RLock()
        self._default_executor = None

        # This is used to do early generation of the c header arguments to release the reference to the dynamic arguments.
        self._generate_c_header_arguments(dynamic_args, dynamic_kwargs)

    @property
    def __mlir__(self):
        """Returns the MLIR code of the JIT-compiled function."""
        return self.artifacts.MLIR if self.artifacts is not None else None

    def generate_execution_args(self, *args, **kwargs):
        return self.args_spec.generate_execution_args(args, kwargs)

    def __call__(self, *args, **kwargs):
        """Executes the jit-compiled function under the currently active CUDA context.

        Calling this method multiple devices is not allowed and will result in unexpected
        CUDA errors. If you need to call the kernel on multiple devices use `to`
        to return a per-device function.
        """
        exe_args, adapted_args = self.args_spec.generate_execution_args(args, kwargs)
        executor = self._default_executor
        if executor is not None:  # Only lock on first call
            return executor.run_compiled_program(exe_args)
        return self.run_compiled_program(exe_args)

    def run_compiled_program(self, exe_args):
        """Executes the jit-compiled function under the currently active CUDA context.

        Calling this method multiple devices is not allowed and will result in unexpected
        CUDA errors. If you need to call the kernel on multiple devices use `to`
        to return a per-device function.
        """
        with self._executor_lock:
            if self._default_executor is None:
                log().debug("Creating default executor.")
                # We use a weak reference here so that this instance does not keep this
                # object alive as it hold a reference to self.
                proxy_self = weakref.proxy(self)
                self._default_executor = proxy_self.to(None)
        return self._default_executor.run_compiled_program(exe_args)

    def _generate_c_header_arguments(self, dynamic_args, dynamic_kwargs):
        """Generates the c header arguments for the AOT C header generation."""
        self.c_header_arguments = None
        from .export import CHeaderArguments

        if dynamic_args is not None or dynamic_kwargs is not None:
            self.dummy_prefix_name = "dummy_prefix_name"
            try:
                # This arguments may be generated failure due to not all the arguments (e.g. custom types) are supported by the AOT C header generator.
                c_header_arguments, packed_args, declarations = (
                    self.export_provider.c_header_generator._generate_arguments(
                        self.dummy_prefix_name,
                        self.args_spec,
                        dynamic_args,
                        dynamic_kwargs,
                    )
                )
                self.c_header_arguments = CHeaderArguments(
                    self.dummy_prefix_name,
                    c_header_arguments,
                    packed_args,
                    declarations,
                    None,
                )
            except Exception as e:
                self.c_header_arguments = CHeaderArguments(
                    self.dummy_prefix_name, [], [], [], str(e)
                )

    def dump_to_object(
        self,
        function_prefix: str,
    ) -> bytes:
        """Dump the compiled ir function to a bytes object with ELF format. The bytes object contains the host
        launch entry function and cubin inside.

        @param function_prefix: The prefix name of the function. This is the user provided unique identifier name of the function to avoid symbol conflict in the generated object file.
        @return: The bytes object of the function.
        """
        # lazy import to avoid circular dependency
        from .export import get_export_module, encode_metadata_into_ir_module

        assert self.export_provider is not None, (
            "Export provider is not set for JitCompiledFunction."
        )
        export_module = get_export_module(self.ir_module, function_prefix)
        export_module = encode_metadata_into_ir_module(
            function_prefix,
            export_module,
            self.args_spec.args_spec,
            self.function_name,
            self.kernel_info,
            self.export_provider.arg_spec_processor,
            self.export_provider.object_file_version,
        )

        cubin_data = None

        def strip_gpu_binary_op(op):
            if op.name == "gpu.binary":
                s = io.BytesIO()
                op.operation.write_bytecode(s)
                nonlocal cubin_data
                cubin_data = s.getvalue()
                cubin_data = cubin_data.split(b'bin = "')[1].split(b'">')[0]
                cubin_data = get_escaped_cubin_bytes(cubin_data)
                op.erase()
                return ir.WalkResult.ADVANCE
            return ir.WalkResult.ADVANCE

        # Strip gpu related to avoid the object file generating builtin module load/unload functions
        export_module.operation.walk(strip_gpu_binary_op)
        cubin_suffix = "cubin"
        if cubin_data is not None:
            cubin_array = array.array("b", cubin_data)
            with (
                export_module.context,
                ir.Location.unknown(),
                ir.InsertionPoint(export_module.body),
            ):
                new_binary_global_op = llvm.GlobalOp(
                    sym_name="_".join([function_prefix, cubin_suffix]),
                    global_type=ir.Type.parse(f"!llvm.array<{len(cubin_array)} x i8>"),
                    linkage=ir.Attribute.parse("#llvm.linkage<external>"),
                    value=ir.DenseIntElementsAttr.get(cubin_array),
                    constant=True,
                )

        if "gpu.container_module" in export_module.operation.attributes:
            del export_module.operation.attributes["gpu.container_module"]
        # Generate the object file

        try:
            with tempfile.NamedTemporaryFile() as tmp_object_file:
                self.export_provider.mlirExecutionEngine.dump_object_file_pic(
                    export_module,
                    tmp_object_file.name,
                    "_".join([function_prefix, self.function_name]),
                )
                with open(tmp_object_file.name, "rb") as f:
                    ret = f.read()
                return ret
        except Exception as e:
            raise DSLRuntimeError(f"Error dumping object file: {e}") from e

    def export_to_c(
        self,
        file_path: str,
        file_name: str,
        function_prefix: str = "",
    ):
        """Exports the jit-compiled function to a C compatible files(header/library).
        This is used for c/cpp AOT support.
        The `file_path` will be used as the directory to save the header and object files.
        The `file_name` will be used as the name of the header and object files. The same file name will always overwrite the existing file.
        The `function_prefix` will be used as the symbol prefix of the generated functions, it is guaranteed by
        the caller that the generated functions are unique.


        The c header file is generated with following components:
        1. The host launch entry function. And the structure definitions of the arguments.
        2. The device metadata load/unload functions.
        3. The cubin data array and len.

        The library contains the binary of the underlying host launch entry function.

        @param jit_function: The jit-compiled function from `cute.compile`.
        @param file_path: The path to the directory where the header and object files will be saved.
        @param file_name: The name of the header and object files.
        @param function_prefix: The prefix of the function. This is the unique identifier name of the function to avoid symbol conflict in the generated object file. Default to the `file_name`.
        """
        if function_prefix is None or function_prefix == "":
            function_prefix = file_name

        assert self.export_provider is not None, (
            "Export provider is not set for JitCompiledFunction."
        )
        # lazy import to avoid circular dependency
        from .export import get_export_module

        export_module = get_export_module(self.ir_module, function_prefix)
        # Generate the c header file
        header_file_content = self.export_provider.c_header_generator(
            function_prefix,
            export_module,
            self.args_spec,
            self.function_name,
            self.kernel_info,
            self.c_header_arguments,
            self.export_provider.dsl._get_dsl().name,
        )
        try:
            with open(os.path.join(file_path, file_name + ".h"), "w") as f:
                f.write(header_file_content)
        except Exception as e:
            raise DSLRuntimeError(f"Error writing header file: {e}") from e

        # Generate the object file
        object_file_content = self.dump_to_object(function_prefix)
        try:
            with open(os.path.join(file_path, file_name + ".o"), "wb") as f:
                f.write(object_file_content)
        except Exception as e:
            raise DSLRuntimeError(f"Error writing object file: {e}") from e
