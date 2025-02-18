# PyCCE

# how to use it
1. cp example/tileop_sample/gen_tileop_func.py .
2. python3 gen_tileop_func.py
3. check the file, the output as shown as follow

custom_op_src
├── custom_add
│   ├── custom_add.h
│   ├── custom_add_pybind.h
│   └── custom_add_tile_op.h
└── custom_op_info.json

4. copy the absolute path of custom_op_info.json to the opcode.cpp.
e.g. RegistorCustomOpThroughJson("custom_op_registry.json") -> RegistorCustomOpThroughJson("your/absolute/path/of/custom_op_info.json")

custom_add.h --> api for custom tensor op
custom_add_pybind.h --> c++ to python pybind api for custom tensor op
custom_add_tile_op.h --> custom tile op header file
custom_op_info.json --> json file for custom tile op information and properties

structure of custom_op_info.json
{
    "customAdd": {
        "op_name": "customAdd",
        "op_src_path": "./custom_op_src/custom_add/custom_add.h",
        "op_pybind_src_path": "./custom_op_src/custom_add/custom_add_pybind.h",
        "tile_op_src_path": "./custom_op_src/custom_add/custom_add_tile_op.h",
        "code_op_code_info": {
            "opcode": 0,
            "core_type": 0,
            "name": "custom_op_code",
            "input_memory_type": [
                0
            ],
            "output_memory_type": [
                0
            ],
            "tile_op_cfg": {
                "name": "custom_tile_op",
                "pipe_id_start": 0,
                "pipe_id_end": 0,
                "core_type": 0
            },
            "op_calc_type": 0,
            "attrs": []
        },
        "input": {
            "tsr2": {
                "name": "tsr2",
                "dtype": "half",
                "shape": 32
            },
            "tsr0": {
                "name": "tsr0",
                "dtype": "half",
                "shape": 32
            },
            "tsr1": {
                "name": "tsr1",
                "dtype": "half",
                "shape": 32
            }
        },
        "output": {}
    }
}


TODO:
- differential input and output
- info for infer shape (elementwise, reduce, broadcast + axis info)
- info for workspace (assign it as an input)
