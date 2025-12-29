## Prerequisites

### macOS

Install required dependencies:

```bash
# Install CMake and pybind11 using Homebrew
brew install cmake
brew install pybind11

# Install Python dependencies (torch and numpy)
pip3 install torch numpy
```

Alternatively, if you prefer using conda:

```bash
# Install CMake and pybind11
brew install cmake
conda install -c conda-forge pybind11

# Install Python dependencies
conda install pytorch numpy -c pytorch
```

## Build and Run Tests

1. Navigate to the prototype directory:
```bash
cd prototype
```

2. Build the project:
```bash
./build.sh
```

This will:
- Configure CMake and build the C++ library
- Build the Python extension module (`pypto_impl`)
- Place the built module in `python/pypto/` directory
- Run all python test cases in python/tests/

## Debug Python tests with C++ breakpoints (VSCode + LLDB)

This project builds a Python extension module via pybind11 (`pypto_impl`). To hit C++ breakpoints while running a Python test, you need to:

- Build with debug symbols (Debug or RelWithDebInfo)
- Attach LLDB to the running Python process

### Prerequisites

- Install VSCode extensions:
  - Python (debugpy)
  - CodeLLDB
- On macOS, allow VSCode to debug other processes:
  - System Settings -> Privacy & Security -> Developer Tools -> enable VSCode

### Build with debug symbols

From `prototype/`:

```bash
./build.sh
```

`build.sh` configures CMake with `-DCMAKE_BUILD_TYPE=Debug` by default.

### VSCode workflow (manual attach)

1. Start Python debugging:
   - Run the VSCode launch config `Python: Debug` (e.g. `prototype/python/tests/test_basic_if_else.py`)
2. Attach LLDB to the same Python process:
   - Run the VSCode launch config `LLDB: Attach to Python (pick process)`
   - Pick the Python process that is running your test
3. Set breakpoints in C++ sources:
   - Breakpoints will be hit for code that is compiled into and loaded by the Python process, such as:
     - `pypto_impl` binding sources (e.g. `src/binding/pypto_impl.cpp`)
     - linked libraries like `ir_proto` / `pass_proto` that are loaded with the extension

### Notes about "any C++ file"

- A breakpoint will stay unverified and never hit if the code is not in the current process address space:
  - the `.cpp` is not built into any target, or
  - the library containing that code is not loaded by Python
- If your C++ code runs during module import/initialization, you may need an "attach window" before it executes (e.g. a short `sleep` in the Python entrypoint) so LLDB can attach in time.
