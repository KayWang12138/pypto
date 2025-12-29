# 18. Serialization(TODO)

PTO-IR supports multiple representation formats: in-memory data structures, human-readable textual format, and compact binary format. This section defines the serialization and deserialization rules for converting between these formats.

---

## 18.1 Format Overview

### 18.1.1 In-Memory Format

The in-memory format is the primary representation used by compilers and optimizers:

* **Data structures**: C++ classes or Python objects
* **Structure**: Hierarchical (Program → Module → Function → Block → Operation)
* **Types**: Strongly typed with full type information
* **Use case**: Compiler internal representation

---

### 18.1.2 Textual Format

Human-readable format for debugging and inspection:

* **Syntax**: MLIR-like textual syntax
* **Readability**: Human-friendly with comments support
* **Use case**: Debugging, documentation, manual editing

**Example:**
```mlir
pto.program.module @main {
  pto.func.func @matmul(%A: tensor<1024x512xf16>, %B: tensor<512x256xf16>)
      -> (tensor<1024x256xf16>) {
    %C = pto.tensor.matmul %A, %B : tensor<1024x512xf16>, tensor<512x256xf16> -> tensor<1024x256xf16>
    pto.func.return %C
  }
}
```

---

### 18.1.3 Binary Format

Compact bytecode format for fast loading and execution:

* **Encoding**: Binary encoding with versioning
* **Size**: Compact representation
* **Speed**: Fast serialization/deserialization
* **Use case**: Runtime loading, distribution, caching

---

## 18.2 Textual Format Specification

### 18.2.1 Syntax Rules

**Identifiers:**
- Function names: `@function_name`
- Block labels: `^bb0`, `^bb1`, ...
- SSA values: `%value0`, `%value1`, ...
- Attributes: `{key = value}`

**Types:**
- Scalar: `i32`, `f16`, `f32`, `bool`
- Tensor: `tensor<shape x element_type>`
- Function: `(T1, T2) -> (R1, R2)`

**Operations:**
```
%result = dialect.operation %operand1, %operand2 
    {attribute = value} 
    : type1, type2 -> result_type
```

---

### 18.2.2 Module Structure

```mlir
pto.program.module @module_name {
  // Global objects
  pto.program.global @global_name : type = value
  
  // Functions
  pto.func.func @function_name(%arg: type) -> (result_type) {
    // Operations
  }
  
  // Entry point
  pto.program.entry @function_name
}
```

---

### 18.2.3 Comments

Textual format supports comments:

```
// Single-line comment
/* Multi-line
   comment */
```

---

### 18.2.4 Core BNF Grammar

This section provides a formal BNF (Backus-Naur Form) grammar for the core textual syntax of PTO-IR. The grammar uses Extended BNF (EBNF) notation:

- `[ ]` denotes optional (0 or 1 occurrence)
- `{ }` denotes repetition (0 or more occurrences)
- `( )` denotes grouping
- `|` denotes alternation (choice)
- `"..."` denotes literal strings
- `<...>` denotes non-terminals

---

#### Top-Level Structure

```
<pto_file>        ::= { <comment> } <module> { <comment> }
<module>          ::= "program.module" <identifier> "{" <module_body> "}"
<module_body>     ::= { <module_item> }
<module_item>     ::= <global> | <function> | <entry> | <launch> | <attr> | <operation> | <comment>
```

---

#### Identifiers and Names

```
<identifier>      ::= <bare_id> | <quoted_id>
<bare_id>         ::= [a-zA-Z_][a-zA-Z0-9_$]*
<quoted_id>       ::= '"' { <escaped_char> | [^"] } '"'
<escaped_char>    ::= "\\" ( "n" | "t" | "\\" | '"' | <unicode> )

<ssa_id>          ::= "%" ( <integer> | <identifier> )
<block_id>        ::= "^" ( <integer> | <identifier> )
<symbol_id>       ::= "@" <identifier>
<integer>         ::= [0-9]+
```

---

#### SSA Values

```
<ssa_use>         ::= <ssa_id>
<ssa_def>        ::= <ssa_id>
<ssa_result>     ::= <ssa_def> | "(" <ssa_def> { "," <ssa_def> } ")"
```

---

#### Types

```
<type>           ::= <scalar_type> 
                   | <tensor_type> 
                   | <tile_type> 
                   | <memref_type>
                   | <shape_type>
                   | <function_type>
                   | <distributed_type>
                   | <pipeline_type>
                   | <instruction_type>
                   | <opaque_type>

<scalar_type>    ::= "i" <width> | "f" <width> | "bf" <width> | "bool" | "index"
<width>          ::= "1" | "8" | "16" | "32" | "64"

<tensor_type>    ::= "tensor<" <shape> "x" <element_type> [ "," <layout> ] ">"
                   | "tensor<*x" <element_type> [ "," <layout> ] ">"

<tile_type>      ::= "tile<" <shape> "x" <element_type> "," <tile_format> ">"

<memref_type>    ::= "memref<" <shape> "x" <element_type> 
                     "," <layout> "," <memory_space> ">"

<shape_type>     ::= "shape<" [ <dim_list> ] ">" | "shape<*>"

<function_type>  ::= "(" [ <type_list> ] ")" "->" "(" [ <type_list> ] ")"

<distributed_type> ::= "dist.tensor<" <dist_strategy> ">"
                     | "dist.group<" <integer> ">"

<pipeline_type>  ::= "pipe.exec.local<" <type> ">"

<instruction_type> ::= "reg." <element_type>
                     | "mma.tile"
                     | "barrier_token"

<opaque_type>    ::= "pto.custom<" <identifier> [ { "," <type> } ] ">"

<shape>          ::= <dim> | <shape> "x" <dim>
<dim>            ::= <integer> | "?" | <symbol>
<symbol>         ::= "%" <identifier>

<dim_list>       ::= <dim> | <dim_list> "," <dim>

<element_type>   ::= <scalar_type>

<layout>         ::= "ND" | "NHWC" | "NCHW" | "NCDHW" | "NDHWC" | "NZ" | "FRACTAL_Z" 
                   | "CUSTOM{" <attribute_dict> "}"

<tile_format>    ::= "ND" | "NZ" | "FRACTAL_Z" | <identifier>

<memory_space>   ::= "DDR" | "L1" | "UB" | "L0A" | "L0B" | "L0C" | "REG" | <identifier>

<dist_strategy>  ::= "replicated" | "sharded" | "partial" | <identifier>

<type_list>      ::= <type> | <type_list> "," <type>
```

---

#### Functions and Blocks

```
<function>       ::= "func.func" <symbol_id> 
                     [ "(" [ <param_list> ] ")" ]
                     [ "->" "(" [ <type_list> ] ")" ]
                     [ <function_attrs> ]
                     "{" <region> "}"

<param_list>     ::= <param> | <param_list> "," <param>
<param>          ::= <ssa_id> ":" <type>

<function_attrs> ::= "{" <attribute_dict> "}"

<region>         ::= { <block> }
<block>          ::= <block_label> [ "(" [ <block_arg_list> ] ")" ] ":" 
                     { <operation> }
                     <terminator>

<block_label>    ::= <block_id>

<block_arg_list> ::= <block_arg> | <block_arg_list> "," <block_arg>
<block_arg>      ::= <ssa_id> ":" <type>

<terminator>     ::= <operation>  // Must be a terminator operation
```

---

#### Operations

```
<operation>      ::= [ <ssa_result> "=" ] 
                     <dialect> "." <op_name>
                     [ <operand_list> ]
                     [ <attribute_dict> ]
                     [ ":" <type_signature> ]
                     [ <region_list> ]

<ssa_result>     ::= <ssa_def> | "(" <ssa_def> { "," <ssa_def> } ")"

<dialect>        ::= "pto." <identifier>
<op_name>        ::= <identifier>

<operand_list>   ::= <ssa_use> | <operand_list> "," <ssa_use>

<type_signature> ::= [ <type_list> ] [ "->" <type_list> ]

<region_list>    ::= <region> | <region_list> <region>
```

---

#### Attributes

```
<attribute_dict> ::= "{" [ <attribute_list> ] "}"
<attribute_list> ::= <attribute> | <attribute_list> "," <attribute>
<attribute>      ::= <identifier> "=" <attribute_value>

<attribute_value> ::= <integer>
                    | <float>
                    | <boolean>
                    | <string>
                    | <symbol_id>
                    | <type>
                    | <attribute_dict>
                    | <attribute_array>
                    | <dense_array>

<attribute_array> ::= "[" [ <attribute_value_list> ] "]"
<attribute_value_list> ::= <attribute_value> 
                          | <attribute_value_list> "," <attribute_value>

<dense_array>    ::= "dense<" <type> ">" "[" <integer_list> "]"
<integer_list>   ::= <integer> | <integer_list> "," <integer>

<boolean>        ::= "true" | "false"
<string>         ::= '"' { <escaped_char> | [^"] } '"'
<float>          ::= [0-9]+ "." [0-9]+ [ "e" [ "+" | "-" ] [0-9]+ ]
```

---

#### Module-Level Constructs

```
<global>         ::= "program.global" <symbol_id> ":" <type> [ "=" <attribute_value> ]
                    [ <attribute_dict> ]

<entry>          ::= "program.entry" <symbol_id>

<launch>         ::= "program.launch" <symbol_id> 
                     [ "on" <launch_attrs> ]
                     [ <attribute_dict> ]

<launch_attrs>   ::= "device" "=" <string> 
                     [ "," "groups" "=" <integer> ]
                     [ "," "threads" "=" <integer> ]

<attr>           ::= "program.attr" <identifier> "=" <attribute_value>
```

---

#### Comments

```
<comment>        ::= "//" { [^\n] } "\n"
                  | "/*" { [^*] | "*" [^/] } "*/"
```

---

#### Grammar Notes

1. **Whitespace**: Whitespace (spaces, tabs, newlines) is generally ignored except where required to separate tokens.

2. **Precedence**: The grammar does not define operator precedence; precedence is determined by the operation semantics defined in each dialect specification.

3. **SSA Semantics**: The grammar enforces SSA form syntactically:
   - Each `<ssa_def>` must be unique within its scope
   - `<ssa_use>` must reference a previously defined `<ssa_def>` or a block argument

4. **Type Inference**: Some operations may omit explicit type signatures when types can be inferred from operands or context.

5. **Dialect Extensions**: Dialect-specific operations follow the general `<operation>` syntax but may have additional constraints defined in their respective dialect specifications.

6. **Regions**: Operations with regions (e.g., `scf.if`, `scf.for`) contain nested `<region>` elements, each with their own SSA scope.

---

#### Example Parse Tree

For the operation:
```mlir
%result = pto.tensor.matmul %A, %B : tensor<1024x512xf16>, tensor<512x256xf16> -> tensor<1024x256xf16>
```

Parse tree structure:
```
<operation>
  ├─ <ssa_result> → <ssa_def> → "%result"
  ├─ <dialect> → "pto.tensor"
  ├─ <op_name> → "matmul"
  ├─ <operand_list>
  │   ├─ <ssa_use> → "%A"
  │   └─ <ssa_use> → "%B"
  └─ <type_signature>
      ├─ <type_list>
      │   ├─ <tensor_type> → "tensor<1024x512xf16>"
      │   └─ <tensor_type> → "tensor<512x256xf16>"
      └─ <type_list>
          └─ <tensor_type> → "tensor<1024x256xf16>"
```

---

## 18.3 Binary Format Specification

### 18.3.1 Encoding Strategy

Binary format uses:

* **Version header**: Format version for compatibility
* **String table**: Shared string pool
* **Type table**: Shared type definitions
* **Operation encoding**: Compact operation representation

---

### 18.3.2 Structure

**Header:**
```
magic_number (4 bytes)
version (2 bytes)
flags (2 bytes)
```

**Sections:**
1. String table
2. Type table
3. Attribute table
4. Operation stream
5. Metadata

---

### 18.3.3 Operation Encoding

Operations encoded as:
- Opcode (variable-length encoding)
- Operand references (indices)
- Result types (type table indices)
- Attributes (attribute table indices)

---

## 18.4 Serialization Rules

### 18.4.1 Type Serialization

**Textual:**
```
tensor<1024x512xf16>
```

**Binary:**
- Type kind identifier
- Shape encoding
- Element type encoding
- Layout encoding (if present)

---

### 18.4.2 Operation Serialization

**Textual:**
```
%result = tensor.matmul %A, %B : tensor<...>, tensor<...> -> tensor<...>
```

**Binary:**
- Dialect identifier
- Operation identifier
- Operand list (SSA value indices)
- Result type list
- Attribute map

---

### 18.4.3 Attribute Serialization

**Textual:**
```
{key = "value", num = 42, flag = true}
```

**Binary:**
- Attribute kind
- Key (string table index)
- Value encoding (type-dependent)

---

## 18.5 Deserialization

### 18.5.1 Validation

Deserialization must:

1. **Verify version**: Check format version compatibility
2. **Validate structure**: Ensure well-formed IR
3. **Check types**: Verify type consistency
4. **Verify SSA**: Ensure SSA correctness

---

### 18.5.2 Error Handling

Deserialization errors:

* **Version mismatch**: Report incompatible version
* **Corrupted data**: Report corruption with location
* **Invalid structure**: Report structural errors
* **Type errors**: Report type inconsistencies

---

## 18.6 Round-Trip Guarantees

Serialization and deserialization should preserve:

* **Semantics**: Program behavior unchanged
* **Structure**: IR structure preserved
* **Types**: All type information preserved
* **Attributes**: All attributes preserved
* **Debug info**: Debug metadata preserved (if enabled)

---

## 18.7 Examples

### Textual to Binary

```mlir
// Textual
pto.func.func @add(%a: i32, %b: i32) -> (i32) {
  %sum = pto.scalar.add %a, %b : i32, i32 -> i32
  pto.func.return %sum
}

// Binary (conceptual)
[FUNC_DECL] [@add] [2 args: i32, i32] [1 result: i32]
  [SCALAR_ADD] [%a, %b] -> [%sum]
  [RETURN] [%sum]
```

---

## 18.8 Summary

Serialization provides:

* Multiple representation formats
* Efficient binary encoding
* Human-readable textual format
* Round-trip preservation guarantees

It enables flexible IR storage, distribution, and debugging.

---

