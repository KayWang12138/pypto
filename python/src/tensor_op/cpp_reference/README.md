This subdirectory contains standalone, out-of-source cpp frontend examples

It serves as reference for line-to-line translation to Python, and for ensuring
the outputs (graph dumps and compiled op binary) are the same as Python-generated output

## Usage

```bash
# first build main ascendcpp as shared lib (same dependency as pypto)
# inside main project dir, not this subdirectory!!
rm -rf build
cmake -B build -DCMAKE_CXX_FLAGS="-fPIC -D_GLIBCXX_USE_CXX11_ABI=0"
cmake --build build -- -j $(nproc)

# Copy config json files into conf/ folder
cd ${ASCENDCPP_DIR}/build/src/
mkdir -p conf && cd conf/
cp ../interface/tile_fwk_config.json .
cp ../passes/tile_fwk_platform_info.json .

# then build standalone cpp reference examples
cd pypto/cpp_reference
rm -rf build
ASCENDCPP_DIR=$(dirname $(dirname $(pwd)))  # point to main repo location
ASCENDCPP_DIR=$ASCENDCPP_DIR cmake -B build
cmake --build build

# run the executables
export GLOBAL_LOG_LEVEL=1  # optionally log all passes
./build/print_dtype
./build/dump_graph
./build/vector_add
./build/manual_record
./build/llama_graph
./build/dynamic_attention
./build/dynamic_loop
# all runs through all passes (last is Pass_31_CodegenPreproc)
```
