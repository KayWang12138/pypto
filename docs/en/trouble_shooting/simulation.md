# SIMULATION Component Error Codes

- **Range**: F9XXXX
- This document describes the error code definitions, scenario descriptions, and troubleshooting recommendations for the SIMULATION component.


## Error Code Definitions and Usage

The unified definitions for related error codes can be found in the `framework/src/cost_model/simulation/utils/simulation_error.h` file.


## Troubleshooting Recommendations

Based on the different ErrorCodes in the logs, refer to the following troubleshooting recommendations:

### EXTERNAL_ERROR External Errors
#### INVALID_CONFIG
1. **Check if the configuration is valid**: Confirm that the passed-in configuration meets length, format, and other requirements.
2. **Check if the configuration exists**: Confirm that the passed-in configuration is a configuration required for simulation.

#### CONFIG_OUT_OF_RANGE
1. Check if the number of configurations exceeds the uint64_t range.

#### INVALID_CONFIG_NAME
1. Check if the configuration name meets requirements. For simulation configuration details, see `framework/src/cost_model/simulation/config/xxx.h`.

#### PERMISSION_CHECK_ERROR
1. Check if the file has read or write permissions.

#### FILE_FORMAT_ERROR
1. **Check if the file format is incorrect**: Confirm that the content of the JSON file meets JSON format requirements.

#### FILE_CONTENT_ERROR
1. Check if the file content meets the agreed-upon requirements.

#### INVALID_PATH
1. Check if the file path is correct.

#### FILE_OPEN_FAILED
1. Confirm if the file has read permissions.
2. Confirm if the file path is correct.
3. Confirm if the file is corrupted.



### FORWARD_SIM

#### INVALID_PIPE_TYPE
1. **Invalid pipe type**: Add a new pipe type to the SCHED_CORE_PIPE_TYPE data structure in `framework/src/cost_model/simulation/common/ISA.h`.

#### INVALID_DATA_TYPE
1. Check if the data type is valid.

#### DEAD_LOCK
1. Find the dot file corresponding to the error under `output/output_xxx/CostModelSimulationOutput/graphs` for analysis.
2. If the root cause still cannot be located, contact the administrator for resolution.

#### FUNC_NOT_SUPPORT
1. Add the new CCE instruction in the GetProgram function in `framework/src/cost_model/simulation_ca/A2A3/model/def.h`.



### PRECISION_SIM

#### NO_SO_EXISTS
1. Check if the .so files required for precision simulation exist and are placed in the correct path.

#### CANN_LOAD_FAILED
1. **Confirm that the CANN environment has been loaded**: `source xxx/set_env.sh`


## General Troubleshooting Recommendations

### 1. Enable Detailed Logging

When encountering SIMULATION component errors, enable detailed logging for more information:

```bash
export ASCEND_GLOBAL_LOG_LEVEL=PYPTO=0 # 0: Debug, 1: Info, 2: Warning, 3: Error (default)
export ASCEND_PROCESS_LOG_PATH=./debug_logs # Specify log persistence path
```

To change PYPTO log output to terminal output, enter the following command:

```bash
export ASCEND_SLOG_PRINT_TO_STDOUT=1 # Change log from persistence to terminal output
```
