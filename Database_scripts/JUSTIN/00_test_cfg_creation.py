#!/usr/bin/env python3
import os
import sys
import subprocess
import r2pipe


def asm_to_dot(asm_file, func_name=None):
    """
    1) Assemble `asm_file` (e.g. A2W.s) into an object (A2W.o).
    2) Use radare2 to auto-analyze the object and export the CFG in DOT format for `func_name`.
       If `func_name` is None, we derive it from the base name of `asm_file`.
    """
    if not os.path.isfile(asm_file):
        raise FileNotFoundError(f"Assembly file '{asm_file}' not found.")

    base_name = os.path.splitext(os.path.basename(asm_file))[0]
    if func_name is None:
        func_name = base_name  # Assume function name matches the file base

    obj_file = base_name + ".o"
    dot_file = base_name + ".dot"

    # Step 1: Assemble the .s -> .o
    gcc_cmd = ["gcc", "-c", asm_file, "-o", obj_file]
    print(f"[DEBUG] Running GCC command: {' '.join(gcc_cmd)}")
    subprocess.run(gcc_cmd, check=True)
    print(f"[DEBUG] Created object file: {obj_file}")

    if not os.path.isfile(obj_file):
        raise RuntimeError(f"Failed to create object file '{obj_file}'.")

    # Step 2: Print symbol table using nm
    print(f"[DEBUG] nm output for {obj_file}:")
    try:
        nm_output = subprocess.check_output(["nm", obj_file], text=True)
        print(nm_output)
    except subprocess.CalledProcessError as e:
        print(f"[WARN] nm failed: {e}")

    # Step 3: Open radare2 (r2pipe) and analyze the object file
    print(f"[DEBUG] Opening radare2 on {obj_file} with function name = {func_name}")
    r2 = r2pipe.open(obj_file, flags=["-2"])  # quiet mode
    print("[DEBUG] Running 'aaa' (auto analyze all)...")
    r2.cmd("aaa")

    # Step 4: List functions (afl) for debugging
    print("[DEBUG] Output of 'afl':")
    afl_output = r2.cmd("afl")
    print(afl_output)

    # Step 5: Attempt to export the CFG in DOT format using the provided function name
    agfd_cmd = f"agfd @ {func_name}"
    print(f"[DEBUG] Running '{agfd_cmd}' ...")
    dot_data = r2.cmd(agfd_cmd)

    # Fallback: if no data, try prefixing with "sym."
    if not dot_data.strip():
        print("[WARN] DOT data is empty. Trying with 'sym.' prefix...")
        agfd_cmd = f"agfd @ sym.{func_name}"
        print(f"[DEBUG] Running '{agfd_cmd}' ...")
        dot_data = r2.cmd(agfd_cmd)

    r2.quit()

    if dot_data.strip() == "":
        print("[ERROR] DOT data is still empty. No CFG was generated for the function.")
    else:
        print(f"[DEBUG] DOT data length: {len(dot_data)} characters")

    # Step 6: Write the DOT data to file
    with open(dot_file, "w") as f:
        f.write(dot_data)
    print(f"[DEBUG] DOT CFG saved to '{dot_file}'")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <assembly_file.s> [function_name]")
        sys.exit(1)

    asm_file = sys.argv[1]
    func_name = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        asm_to_dot(asm_file, func_name)
        print("[INFO] Finished generating the .dot file!")
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
