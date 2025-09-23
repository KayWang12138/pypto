.PHONY: copy_json copy_libs build_pypackage

clean:
	rm -rf pto/*.so pto/*.json

# Copy all AscendC++ Json files inside pypackage
copy_json:

# Copy all AscendC++ dynlibs inside pypackage
copy_libs:
	cp ../../../build/src/operator/libtile_fwk_operator.so pto/
	cp ../../../build/src/interface/libtile_fwk_interface.so pto/
	cp ../../../build/src/passes/libtile_fwk_passes.so pto/
	cp ../../../build/src/machine/libtile_fwk_compiler.so pto/
	cp ../../../build/src/codegen/libtile_fwk_codegen.so pto/
	cp ../../../build/src/cost_model/simulation/libtile_fwk_simulation.so pto/
	cp ../../../build/src/cost_model/simulation_ca/libtile_fwk_simulation_ca.so pto/
	cp ../../../build/tests/runtime/libtile_fwk_runtime.so pto/

install_pypackage: copy_json copy_libs
	cp ../../../build/python/src/tensor_op/pto_impl*.so pto/
	pip3 install -v --force-reinstall .
