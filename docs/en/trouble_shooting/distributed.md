# DISTRIBUTED Component Error Codes

- **Range**: FAXXXX
- This document describes the error code definitions, scenario descriptions, and troubleshooting recommendations for the DISTRIBUTED component.

## Error Code Definitions

The unified definitions for related error codes can be found in the `framework/src/codegen/utils/distributed_error.h` file.

---

## Troubleshooting Recommendations

Based on the different ErrorCodes in the logs, refer to the following troubleshooting recommendations:

### INVALID_GROUP_NAME

1. **Check if the communication group name is empty**: Confirm that the passed-in group_name is not nullptr and is not an empty string.
2. **Check the communication group name length range**: Confirm that the passed-in group_name length is within the range [1, 128), avoiding lengths that are too long or invalid.

### INVALID_WORLD_SIZE

1. **Check the communication group creation**: Ensure that the total number of processes passed in when calling create_shmem_tensor is not 0. When calling create_shmem_tensor repeatedly, ensure that the passed-in group_name is consistent.

### INVALID_TENSOR_DIM

1. **Check tensor dimensions**: Confirm that the tensor dimension is 2, meeting the requirements.

### INVALID_TENSOR_SHAPE

1. **Check tensor shape**: Ensure that the tensor shape is 2-dimensional and that the size of each dimension is a positive integer, avoiding zero or negative dimensions.
2. **Check tensor shape validity**: Confirm that the shape of each dimension meets expectations and satisfies the requirements for subsequent operations.

### INVALID_TENSOR_DTYPE

1. **Check tensor type**: Confirm that the tensor does not contain low-precision or high-precision types unsupported by the current hardware, and that the data type meets expectations.

### INVALID_TENSOR_FORMAT

1. **Check tensor format**: Confirm that the tensor format is ND format, ensuring the data format complies with specifications.

### INVALID_SHMEM_TENSOR

1. **Check input Shmem Tensor**: Determine the cause based on the error message. The possible cause is that the ShmemTensor does not contain a valid data or signal Tensor.

### INVALID_SHMEM_VIEW_PARAM

1. **Check ShmemView interface parameters**: Determine the cause based on the error message. The possible cause is that the shape or offset information passed to ShmemView is invalid.

### INVALID_OP_TYPE

1. **Check ShmemWaitUntil**: Determine the cause directly based on the error message. The possible cause is that an unsupported comparison type was passed to the ShmemWaitUntil interface.

### INVALID_OPERAND_NUM

1. **Check the number of input/output parameters**: Ensure that the number of passed-in input and output parameters is consistent with the API definition.

### INVALID_TILE_DIM

1. **Check tile dimensions**: Confirm that the tile dimension is 2.

### INVALID_TILE_SHAPE

1. **Check tile dimensions**: Confirm that each dimension value of the tile must be greater than 0 and are all valid values.

### INVALID_ALIGNMENT

1. **Check the total byte size of the UB buffer**: Ensure that the total byte size of the UB buffer is an integer multiple of 256 bytes.

### WIN_SIZE_EXCEED_LIMIT

1. **Check the created shmem_tensor**: Confirm that the total element size of all created shmem_tensors is less than 1024 * 1024 * 200 bytes.

### TILE_NUM_EXCEED_LIMIT

1. **Check the total number of tiles**: Confirm that the total number of tiles set before calling shmem_wait_until does not exceed 1024.

### DIVISION_BY_ZERO

1. **Check tile validity**: Confirm that there are no illegal zero values for tile sizes in any dimension.

### AICPU_TASK_TIMEOUT

1. **AICPU wait timeout**: Confirm that the signal sent by shmem_signal can be normally received by shmem_wait_until and waits for completion.
2. **Check log context**: Refer to the `docs\trouble_shooting\machine.md` file to enable DEBUG logging.

### AICPU_TASK_NUM_EXCEED_LIMIT

1. **Check task queue size**: Confirm that the number of tasks in the SignalTileOp queue has not exceeded the maximum capacity limit.
2. **Check task count limit**: Confirm that the current task count (taskCount) does not exceed 1024.
3. **Check log context**: Refer to the `docs\trouble_shooting\machine.md` file to enable DEBUG logging.

### AICPU_TASK_QUEUE_EMPTY

1. **Check the AICPU task queue**: Confirm that the task queue is not empty before executing tasks, avoiding task operations on an empty queue.
2. **Check log context**: Refer to the `docs\trouble_shooting\machine.md` file to enable DEBUG logging.

### AICPU_TASKID_NOT_IN_MAP

1. **Check task ID**: Confirm that the given taskId exists in the task ID mapping table, avoiding task ID lookup failures.
2. **Check log context**: Refer to the `docs\trouble_shooting\machine.md` file to enable DEBUG logging.

### INVALID_GROUP_INDEX

1. **Check communication group index**: Confirm that the given groupIndex is less than the total number of communication groups commGroupNum_, ensuring the index is within the valid range.
2. **Check log context**: Refer to the `docs\trouble_shooting\machine.md` file to enable DEBUG logging.

### NULLPTR

1. **Check runtime management objects**: Ensure that runtime dependency objects such as AicoreManager have been correctly initialized and passed in.
2. **Check log context**: Refer to the `docs\trouble_shooting\machine.md` file to enable DEBUG logging.
