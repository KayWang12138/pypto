# 17. Platform Abstractions Dialect

The `pto.platform` dialect models the execution environment, including execution units, memory hierarchy, interconnections, and distributed structures. These abstractions are used by transformation passes to make informed decisions about optimization, scheduling, and resource allocation. It is also used by costmodel, codegen, and other modules that involve processing hardware information.

---

## 17.1 Purpose

The core goal of Platform IR is to **turn “platform hardware information” into structured IR objects**, so that it becomes:

- **Serializable**: platform information can be saved, replayed, and diffed together with IR.
- **Verifiable**: consistency checks can be performed at IR level (e.g., counts, path legality, missing versions).
- **Queryable**: Codegen / Machine / CostModel share a unified query surface, without relying on external environment or implicit global state.

Platform information typically comes from runtime probing or INI configuration, but in the IR semantics it is represented as `pto.platform.*` dialect objects.

---

## 17.2 Dialect Characteristics

### 17.2.1 Meta-Dialect Nature

`pto.platform` is a meta-dialect similar to `pto.debug` and `pto.config`:
* Does not participate in lowering flows
* Attaches to program or module level
* Used by transformation passes for querying hardware characteristics
* Can be queried and transformed like other IR operations

### 17.2.2 Platform Modeling

Platform abstractions model:
* **Execution units**: Compute units (vector core, cube core, aicpu.) with throughput, latency, and capabilities
* **Memory hierarchy**: Memory levels (L0, L1, UB, DDR) with size, bandwidth, and latency. Cache with size, latency, and eviction policies
* **Interconnections**: On-chip and off-chip interconnects with bandwidth and topology
* **Distributed structure**: NUMA domains, clusters, multi dies

---

## 17.3 Types and Enums

### 17.3.1 NPU Architecture

Text forms:

- `#pto.platform.arch<DAV_1001>`
- `#pto.platform.arch<DAV_2201>`
- `#pto.platform.arch<DAV_3510>`
- `#pto.platform.arch<UNKNOWN>`

Semantics:

- `DAV_1001`: Ascend910
- `DAV_2201`: Ascend910B / Ascend910_93
- `DAV_3510`: Ascend910_95
- `UNKNOWN`: unrecognized

> Note: this mapping must stay consistent with the string mapping of `Short_SoC_version` in INI (see Section 17.7).

### 17.3.2 MemoryType

Text forms:

- `#pto.platform.mem<UB>`
- `#pto.platform.mem<L1>`
- `#pto.platform.mem<L0A>`
- `#pto.platform.mem<L0B>`
- `#pto.platform.mem<L0C>`
- `#pto.platform.mem<DDR>` (device-side DDR)
- `#pto.platform.mem<UNKNOWN>`

---

## 17.4 Attributes

### 17.4.1 Version Attributes

- `core.version`: string, e.g., `"AIC_C_320"`
- `core.ccec_version`: string, e.g., `"ccec-xxx"`

### 17.4.2 Memory Attributes

- `mem.size`: `i64` (or `index`), capacity (unit is platform-defined; typically bytes)
- `mem.valid_size`: optional `i64`, usable capacity (reserved for future integration)

> Note: Die-level memory limits are **derived query results** computed by `Die::GetMemoryLimit(MemoryType)` from unit-owned `pto.platform.mem` objects (see Section 17.6.1).

---
---

## 17.5 Operations

### 17.5.1 pto.platform.platform

**Semantics**: root object of platform description, recommended to be a symbol operation.

**Text format**:

```mlir
pto.platform.platform @<name> attributes {
  cluster_num = <i64>,
  host_num    = <i64>
} {
  pto.platform.cluster ...
}
```

**Constraints**:

- `@<name>` must be unique in the module symbol table.
- `cluster_num >= 0`, `host_num >= 0`

---

### 17.5.2 pto.platform.cluster

**Semantics**: Cluster level, describing the SoC group and count.

**Text format**:

```mlir
pto.platform.cluster attributes { soc_num = <i64> } {
  pto.platform.soc ...
}
```

**Constraints**:

- `soc_num >= 0`

---

### 17.5.3 pto.platform.soc

**Semantics**: SoC level, describing NPUArch, die count, various core/cpu counts, and core version info.

**Text format**:

```mlir
pto.platform.soc attributes {
  arch            = #pto.platform.arch<...>,
  dies_num        = <i64>,
  ai_cpu_num      = <i64>,
  ai_core_num     = <i64>,
  cube_core_num   = <i64>,
  vector_core_num = <i64>
} {
  pto.platform.version ...
  pto.platform.die ...
}
```

**Constraints**:

- `dies_num >= 0`
- `ai_cpu_num/ai_core_num/cube_core_num/vector_core_num >= 0`
- `cube_core_num <= ai_core_num` (if the platform semantics require cube cores to be a subset of AI cores; otherwise this can be relaxed)

---

### 17.5.4 pto.platform.die

**Semantics**: Die level, the main carrier of **data paths**, and contains lower-level structures such as corewrap/aicpu.  
**Note**: Die-level memory limits are **not stored** on `pto.platform.die`. They are **derived query results** computed by `Die::GetMemoryLimit(MemoryType)` from unit-owned `pto.platform.mem` objects under cores (see Section 17.6.1).

**Text format**:

```mlir
pto.platform.die {
  pto.platform.mem.path #pto.platform.mem<DDR> -> #pto.platform.mem<UB>
  ...
  pto.platform.corewrap ...
}
```

**Constraints**:

- `from != to` for `mem.path`

---

### 17.5.5 pto.platform.corewrap

**Semantics**: CoreWrap level, binds AIC/AIV cores into the same assembly unit and expresses the quantity ratio.

**Text format**:

```mlir
pto.platform.corewrap attributes {
  aic_num = <i64>,
  aiv_num = <i64>
} {
  // Define or reference the two core descriptors in this CoreWrap
  pto.platform.core @aic attributes { kind = "AIC", ... }
  pto.platform.core @aiv attributes { kind = "AIV", ... }
}
```

**Constraints**:

- `aic_num >= 0`, `aiv_num >= 0`
- `kind` must be `"AIC"` or `"AIV"`

---

### 17.5.6 pto.platform.core

**Semantics**: Core descriptor, carrying core-level version and (optional) feature fields. AIC/AIV frequency, register widths, FixPipe support, etc. can be attached as extensible attributes here.  
Unit-owned memory capacities are represented by nested `pto.platform.mem` objects under the core.

**Text format**:

```mlir
pto.platform.core @aic attributes {
  kind         = "AIC",
  version      = "...",
  ccec_version = "...",
  cube_freq    = <i64>,          // AIC extension
  fixpipe      = <bool>          // AIC extension
} {
  pto.platform.mem #pto.platform.mem<L0A> attributes { size = <i64> }
  pto.platform.mem #pto.platform.mem<L0B> attributes { size = <i64> }
  pto.platform.mem #pto.platform.mem<L0C> attributes { size = <i64> }
}
```

```mlir
pto.platform.core @aiv attributes {
  kind                = "AIV",
  version             = "...",
  ccec_version        = "...",
  vec_freq            = <i64>,   // AIV extension
  vector_reg_width    = <i64>,
  predicate_reg_width = <i64>,
  wide_reg_width      = <i64>
} {
  pto.platform.mem #pto.platform.mem<L1> attributes { size = <i64> }
  pto.platform.mem #pto.platform.mem<UB> attributes { size = <i64> }
}
```

**Constraints**:

- `kind` must match the extension fields: e.g., `cube_freq/fixpipe` are meaningful only for AIC.
- capacities in `mem.size` must be `>= 0`

---

### 17.5.7 pto.platform.mem

**Semantics**: declares a memory descriptor owned by an execution unit (e.g., a core), storing capacity information.

**Text format**:

```mlir
pto.platform.mem <mem_type> attributes { size = <i64> }
```

---

### 17.5.8 pto.platform.mem.path

**Semantics**: declares a data-movement path (directed edge).

**Text format**:

```mlir
pto.platform.mem.path <mem_type> -> <mem_type>
```

---

### 17.5.9 pto.platform.version

**Semantics**: declares a version bundle for SoC or cores (useful when version info comes from external config and needs a centralized representation).

**Text format**:

```mlir
pto.platform.version attributes {
  aic_version      = "...",
  aiv_version      = "...",
  aic_ccec_version = "...",
  aiv_ccec_version = "..."
}
```

---

### 17.5.10 pto.platform.inst (Reserved)

**Semantics**: intended to model instruction entries (intrinsic name, variants, category, cost counters, etc.).  
Once instruction resource tables are integrated, `pto.platform.inst` can carry the data and bind to cores.

---

### 17.5.11 pto.platform.pipe (Reserved)

**Semantics**: intended to model Pipe (pipeline) information.

---

## 17.6 Semantics and Verification Rules

### 17.6.1 Unified memory-limit lookup rule (AIC preferred / AIV fallback)

Die-level memory limits are **derived query results** computed by `Die::GetMemoryLimit(MemoryType)` from unit-owned `pto.platform.mem` objects inside `pto.platform.corewrap`.

For a given `mem_type`:

- If both AIC and AIV are missing for the same `mem_type`, then `limit = 0`
- If AIC is missing and AIV exists, then `limit = AIV.mem.size[mem_type]`
- If AIC exists, then `limit = AIC.mem.size[mem_type]`

**Typical usage** (pseudo-code; illustrative only):

```text
die = platform.cluster.soc.die
l0a_limit = GetMemoryLimit(die, L0A)
ub_limit  = GetMemoryLimit(die, UB)
```

---

### 17.6.2 Data path legality

- `pto.platform.mem.path from -> to` forbids `from == to`
- Partial paths are allowed; but if `from/to` are declared, both must be known `MemoryType` values

### 17.6.3 Version field consistency

- If present, `version/ccec_version` of AIC/AIV cores should be non-empty strings
- `SoC.arch` should not be `UNKNOWN` unless the platform is explicitly in “unknown/placeholder platform” mode

---

## 17.7 Field Alignment with INI

This section aligns IR fields with INI configuration fields, for consistent IR generation and optional INI round-tripping validation.

### 17.7.1 INI snippet example

```ini
[version]
Short_SoC_version=Ascend910B
; -> pto.platform.soc.arch = #pto.platform.arch<DAV_2201>

[SoCInfo]
ai_core_cnt=32
; -> pto.platform.soc.ai_core_num = 32
cube_core_cnt=32
; -> pto.platform.soc.cube_core_num = 32
vector_core_cnt=32
; -> pto.platform.soc.vector_core_num = 32
ai_cpu_cnt=8
; -> pto.platform.soc.ai_cpu_num = 8

[AICoreSpec]
l0_a_size=65536
; -> pto.platform.core(kind="AIC").mem(#pto.platform.mem<L0A>).size = 65536
l0_b_size=65536
; -> pto.platform.core(kind="AIC").mem(#pto.platform.mem<L0B>).size = 65536
l0_c_size=131072
; -> pto.platform.core(kind="AIC").mem(#pto.platform.mem<L0C>).size = 131072
l1_size=1048576
; -> pto.platform.core(kind="AIV").mem(#pto.platform.mem<L1>).size  = 1048576
ub_size=262144
; -> pto.platform.core(kind="AIV").mem(#pto.platform.mem<UB>).size  = 262144
```

### 17.7.2 Mapping table

| INI section | key / source | IR field |
|---|---|---|
| `[version]` | `Short_SoC_version` | `pto.platform.soc.arch` |
| `[version]` | `GetCoreVersion({"AIC","AIV"})` | `pto.platform.core(kind="AIC/AIV").version` |
| `[version]` | `GetCCECVersion({"AIC","AIV"})` | `pto.platform.core(kind="AIC/AIV").ccec_version` |
| `[SoCInfo]` | `ai_core_cnt` | `pto.platform.soc.ai_core_num` |
| `[SoCInfo]` | `cube_core_cnt` | `pto.platform.soc.cube_core_num` |
| `[SoCInfo]` | `vector_core_cnt` | `pto.platform.soc.vector_core_num` |
| `[SoCInfo]` | `ai_cpu_cnt` | `pto.platform.soc.ai_cpu_num` |
| `[AICoreSpec]` | `l0_a_size/l0_b_size/l0_c_size` | `pto.platform.core(kind="AIC").mem(L0A/L0B/L0C).size` |
| `[AICoreSpec]` | `l1_size/ub_size` | `pto.platform.core(kind="AIV").mem(L1/UB).size` |
| `GetDataPath([.])` | `(from,to)` | `pto.platform.mem.path from -> to` |

---

## 17.8 Examples

> Note: The MLIR snippets in this section are **illustrative pseudo-IR** to show hierarchy/fields.
> The exact assembly format may differ in the real implementation.

### 17.8.1 Minimal Platform Description

```mlir
pto.platform.platform @ascend910b attributes { cluster_num = 1, host_num = 1 } {
  pto.platform.cluster attributes { soc_num = 1 } {
    pto.platform.soc attributes {
      arch            = #pto.platform.arch<DAV_2201>,
      dies_num        = 1,
      ai_cpu_num      = 8,
      ai_core_num     = 32,
      cube_core_num   = 32,
      vector_core_num = 32
    } {
      pto.platform.die {
        // Data paths
        pto.platform.mem.path #pto.platform.mem<DDR> -> #pto.platform.mem<UB>
        pto.platform.mem.path #pto.platform.mem<L0C> -> #pto.platform.mem<DDR>

        // CoreWrap lives under Die (matches the Platform/SoC/Die/CoreWrap hierarchy)
        pto.platform.corewrap attributes { aic_num = 1, aiv_num = 1 } {
          pto.platform.core @aic attributes { kind="AIC", version="AIC_C_320", ccec_version="ccec-x" } {
            pto.platform.mem #pto.platform.mem<L0A> attributes { size = 65536 }
            pto.platform.mem #pto.platform.mem<L0B> attributes { size = 65536 }
            pto.platform.mem #pto.platform.mem<L0C> attributes { size = 131072 }
          }
          pto.platform.core @aiv attributes { kind="AIV", version="AIV_320",  ccec_version="ccec-y" } {
            pto.platform.mem #pto.platform.mem<L1> attributes { size = 1048576 }
            pto.platform.mem #pto.platform.mem<UB> attributes { size = 262144 }
          }
        }
      }
    }
  }
}
```

### 17.8.2 Referencing Platform in pto.program.module

```mlir
pto.program.module @main attributes { pto.platform.target = @ascend910b } {
  // Platform description can live in the same module (or in an outer container)
  pto.platform.platform @ascend910b attributes { cluster_num = 1, host_num = 1 } {
    pto.platform.cluster attributes { soc_num = 1 } {
      pto.platform.soc attributes {
        arch            = #pto.platform.arch<DAV_2201>,
        dies_num        = 1,
        ai_cpu_num      = 8,
        ai_core_num     = 32,
        cube_core_num   = 32,
        vector_core_num = 32
      } {
        pto.platform.die {
          pto.platform.mem.path  #pto.platform.mem<DDR> -> #pto.platform.mem<UB>

          pto.platform.corewrap attributes { aic_num = 1, aiv_num = 1 } {
            pto.platform.core @aic attributes { kind="AIC", version="AIC_C_320", ccec_version="ccec-x" } {
              pto.platform.mem #pto.platform.mem<L0C> attributes { size = 131072 }
            }
            pto.platform.core @aiv attributes { kind="AIV", version="AIV_320",  ccec_version="ccec-y" } {
              pto.platform.mem #pto.platform.mem<UB> attributes { size = 262144 }
            }
          }
        }
      }
    }
  }

  // Business IR is decoupled from the platform: the platform is only module-level target metadata
  pto.program.global @W : tensor<1024x1024xf16> = { ... }
  pto.program.entry @matmul_kernel

  pto.func.func @matmul_kernel(%A: tensor<1024x1024xf16>,
                               %B: tensor<1024x1024xf16>)
      -> (tensor<1024x1024xf16>)
  {
      statement.block {
        %C = pto.tensor.matmul %A, %B
          : tensor<1024x1024xf16>, tensor<1024x1024xf16> -> tensor<1024x1024xf16>
      }
      statement.return %C
  }
}
```

### 17.8.3 Data Paths (Nearest Path) Example

```mlir
pto.platform.die {
  // Declare explicit path edges
  pto.platform.mem.path #pto.platform.mem<DDR> -> #pto.platform.mem<UB>
  pto.platform.mem.path #pto.platform.mem<UB>  -> #pto.platform.mem<L1>
  pto.platform.mem.path #pto.platform.mem<L1>  -> #pto.platform.mem<L0C>

  // Semantics: NearestPath(DDR, L0C) may return [DDR, UB, L1, L0C]
  // Note: NearestPath is a query semantic; it does not have to correspond to a dedicated IR op.
}
```

---
