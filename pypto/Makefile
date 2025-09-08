.PHONY: clean build test

clean:
	rm -f pto*.so
	rm -rf build

build: clean
	export pybind11_DIR=$(shell python -m site --user-site)/pybind11 \
	export ASCENDCPP_DIR=$(shell pwd)/../ \
	&& cmake -B build && cmake --build build

test:
	pytest test

test_examples:
	python example/dynamic_loop_nomacro.py
	python example/record_loop_func.py
	python example/record_func_static.py
	python example/vector_op.py
	python example/matrix_op.py
