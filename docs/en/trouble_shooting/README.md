# PYPTO Framework Error Code Scheme

This document describes the overall range and conventions for error codes across PYPTO components. For specific error code definitions and troubleshooting guidance for each component, see the links below.

## Overall Range

All ASSERT ERROR reports must be carried by error codes. The error code ranges for each component are as follows:

| Range            | Component/Purpose |
|------------------|-------------------|
| F1XXXX           | External restrictions (user syntax issues, scenario limitations, version compatibility, etc.) |
| F2-F3XXXX        | FUNCTION      |
| F4-F5XXXX        | PASS          |
| F6XXXX           | CODEGEN       |
| F7-F8XXXX        | MACHINE       |
| F9XXXX           | SIMULATION    |
| FAXXXX           | DISTRIBUTED   |
| FBXXXX           | VERIFY        |
| FCXXXX           | OPERATION     |
| FC0-FC2XXX       | VECTOR        |
| FC3-FC5XXX       | MATMUL        |
| FC6-FC8XXX       | CONV          |
| FC9XXX           | View Operators |

## Component Error Code Documentation

| Component   | Range      | Documentation |
|-------------|------------|---------------|
| FUNCTION    | F2-F3XXXX  | [function.md](function.md) |
| PASS        | F4-F5XXXX  | [pass.md](pass.md) |
| CODEGEN     | F6XXXX     | [codegen.md](codegen.md) |
| MACHINE     | F7-F8XXXX  | [machine.md](machine.md) |
| SIMULATION  | F9XXXX     | [simulation.md](simulation.md) |
| DISTRIBUTED | FAXXXX     | [distributed.md](distributed.md) |
| VERIFY      | FBXXXX     | [verify.md](verify.md) |
| OPERATION   | FCXXXX     | [operation.md](operation.md) |
| VECTOR      | FC0-FC2XXX | [vector.md](vector.md) |
| MATMUL      | FC3-FC5XXX | [matmul.md](matmul.md) |
| CONV        | FC6-FC8XXX | [conv.md](conv.md) |
| View Operators | FC9XXX  | [view_op.md](view_op.md) |

## Principles

- **Error Reporting and Logging**: All ASSERT/CHECK/ERROR reports in each component must use the error codes from that component's error header file, and the code must be carried in the log or exception.
- **Component-Based Management**: Error code header files for each component are maintained within their respective component directories and are not centralized into a single header file.
- **Documentation Supplement**: If the ErrorMsg alone cannot explain the cause or makes it difficult to locate the issue, a trouble_shooting document (cause, troubleshooting steps, and solution) must be added for that error code.
- **Associated Skill**: Each component's error code documentation may annotate each error code with an **Associated Skill**, pointing to the corresponding skill under [.agents/skills](../../.agents/skills) (e.g., `pypto-environment-setup`), to facilitate loading that skill during troubleshooting for environment diagnosis, operator development, or performance tuning.
- **From High-Level to Sub-Process**: Error code classification should first organize high-level processes (Category), then refine sub-processes (Scene); enumerations in the header file should be organized in this order.
- **Single Responsibility**: One error code corresponds to only one scenario (one type of problem), with a single semantic meaning for quick root cause identification by code value.

## Usage Example (Using MACHINE as an Example)

When reporting an error or logging ERROR, include the error code — it will appear in the log as **Fxxxxx**. The enum value is the error code and can be directly converted to a numeric value. For example, the original log:

```cpp
MACHINE_LOGE("xxx failed.");
```

Should be changed to the format with an error code (using `MACHINE_LOGE`, enum value printed via mask as **Fxxxxx**):

```cpp
#include "tilefwk/pypto_fwk_log.h"

MACHINE_LOGE(SchedErr::HANDSHAKE_TIMEOUT, "hand shake timeout.");
```

Effect:
[ERROR] PYPTO(411657):2026-03-13 14:25:42.400 [host_machine.cpp:135][MACHINE]:ErrCode: F72002! hand shake timeout.
