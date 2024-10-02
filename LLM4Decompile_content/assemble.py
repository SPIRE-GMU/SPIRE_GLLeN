import subprocess
import os

OPT = ["O0", "O1", "O2", "O3"]
fileName = "passwd"  #'path/to/file'
func0 = "generate_password"
for opt_state in OPT:
    output_file = fileName + "_" + opt_state
    input_file = fileName + ".c"
    compile_command = f"gcc -o {output_file}.o {input_file} -{opt_state} -lm"  # compile the code with GCC on Linux
    subprocess.run(compile_command, shell=True, check=True)
    compile_command = f"objdump -d {output_file}.o > {output_file}.s"  # disassemble the binary file into assembly instructions
    subprocess.run(compile_command, shell=True, check=True)

    input_asm = ""
    with open(output_file + ".s") as f:  # asm file
        asm = f.read()
        if (
            "<" + func0 + ">:" not in asm
        ):  # IMPORTANT replace func0 with the function name
            raise ValueError("compile fails")
        asm = (
            "<" + func0 + ">:" + asm.split("<" + func0 + ">:")[-1].split("\n\n")[0]
        )  # IMPORTANT replace func0 with the function name
        asm_clean = ""
        asm_sp = asm.split("\n")
        for tmp in asm_sp:
            if len(tmp.split("\t")) < 3 and "00" in tmp:
                continue
            idx = min(len(tmp.split("\t")) - 1, 2)
            tmp_asm = "\t".join(tmp.split("\t")[idx:])  # remove the binary code
            tmp_asm = tmp_asm.split("#")[0].strip()  # remove the comments
            asm_clean += tmp_asm + "\n"
    input_asm = asm_clean.strip()
    before = f"# This is the assembly code:\n"  # prompt
    after = "\n# What is the source code?\n"  # prompt
    input_asm_prompt = before + input_asm.strip() + after
    with open(fileName + "_" + opt_state + ".asm", "w", encoding="utf-8") as f:
        f.write(input_asm_prompt)
