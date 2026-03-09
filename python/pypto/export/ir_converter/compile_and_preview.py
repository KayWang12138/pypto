from pypto_ir import ir
import os

def compile_and_preview(program: ir.Program, title: str, strategy, backend_type):
    print(f"\n=== {title} ===")
    print("[Frontend IR Preview]")
    print("-" * 60)
    print("\n".join(ir.python_print(program).splitlines()[:120]))
    print("-" * 60)

    out_dir = ir.compile(program, dump_passes=True, strategy=strategy, backend_type=backend_type)
    print("Artifacts written to:", out_dir)

    pto_file = os.path.join(out_dir, "output.pto")
    if os.path.exists(pto_file):
        with open(pto_file, "r") as f:
            pto = f.read()
        print("\n[PTO Preview]")
        print("=" * 60)
        print("\n".join(pto.splitlines()[:160]))
        print("=" * 60)
    else:
        print("No output.pto found in:", out_dir)
