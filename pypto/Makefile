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
