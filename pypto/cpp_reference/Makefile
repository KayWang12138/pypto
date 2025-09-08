.PHONY: clean build test

clean:
	rm -rf build

build: clean
	export ASCENDCPP_DIR=$(shell pwd)/../../ && cmake -B build && cmake --build build

test:
	./build/print_dtype
	./build/dump_graph
	./build/vector_add
	./build/manual_record
	./build/llama_graph
	./build/dynamic_attention
	./build/dynamic_loop
