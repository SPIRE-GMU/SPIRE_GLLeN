"""
This file iterates through the exebench
dataset and tests if LLM4Decompile is able to 
decompile a recombilable C file
"""

from datasets import load_dataset
import time
import gc
import os
import shutil
import subprocess
import torch
import pprint
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
import argparse

MODEL_PATH = "/home/spire2/LLM4Decompile/llm4decompile-22b-v2"


origin_no_path = "origin"

recompile_no_path = "new_file"
optimization = ["O1", "O2", "O3"]


def main():
    """
    The main function controling all other functions
    """
    currently_a_success = True
    garbage_file_count_1 = 0
    garbage_file_count_2 = 0
    garbage_file_count_3 = 0

    success_count_1 = 0
    success_count_2 = 0
    success_count_3 = 0

    # 1) Load dataset split. In this case, synthetic test split
    dataset = load_dataset(
        "jordiae/exebench", split="train_real_compilable"
    )  # , use_auth_token=True)
    # 2) Iterate over dataset
    tokenizer, model = load_model()
    for row in dataset:
        for opt in range(len(optimization)):
            recompiled_file = "new_file" + optimization[opt] + ".c"
            original_file = "origin" + optimization[opt] + ".c"
            try:
                func0 = row["fname"]
                # pprint.pp(row)
                with open(original_file, "w") as f:
                    f.write(
                        "#include <stdint.h>\n#include <stdio.h>\n"
                        + row["real_deps"]
                        + "\n"
                        + row["func_def"]
                    )
                f.close()
                #    row = 0
                #    while row < 1:
                # for row in dataset:
                # Do stuff with each row
                # 3) Option A: Manually access row fields. For instance, access the function definition:
                # pprint.pp(dataset[row])
                # print('*****')
                # print(row['func_def'])  # print function definition
                # print('*****')
                # print(row['asm']['code'][0])  # print assembly with the first target, angha_gcc_x86_O0
                # print first I/O example (synthetic)
                # print('Input:', exebench_dict_to_dict(row['synth_io_pairs']['input'][0]))
                # print('Output:', exebench_dict_to_dict(row['synth_io_pairs']['output'][0]))
                # print(row['synth_exe_wrapper'][0])  # print C++ wrapper to run function with IO
                # You can manually compile, run with IO, etc
                #       row += 1

                asm_file = assemble(
                    original_file, origin_no_path, func0, optimization[opt]
                )

                decompiled_func = decompiler(asm_file, tokenizer, model)

                with open(recompiled_file, "w") as d:
                    d.write(
                        "#include <stdint.h>\n#include <stdio.h>\n"
                        + row["real_deps"]
                        + "\n"
                        + decompiled_func
                    )
                d.close()

                currently_a_success = True

                # Clear the terminal screen

                # Display the ASCII art

                # Get a list of files with .c or .o extensions in the current working directory

            except:
                print("This file sucks becuse exebench sucks")
                currently_a_success = False
            try:
                if currently_a_success:
                    if opt == 0:
                        assemble(
                            recompiled_file, recompile_no_path, func0, optimization[opt]
                        )
                        shutil.copy(
                            original_file,
                            f"red-team-success/{origin_no_path}{optimization[opt]}{success_count_1}.c",
                        )
                        shutil.copy(
                            recompiled_file,
                            f"red-team-success/{recompile_no_path}{optimization[opt]}{success_count_1}.c",
                        )

                        success_count_1 += 1
                    elif opt == 1:
                        assemble(
                            recompiled_file, recompile_no_path, func0, optimization[opt]
                        )
                        shutil.copy(
                            original_file,
                            f"red-team-success/{origin_no_path}{optimization[opt]}{success_count_2}.c",
                        )
                        shutil.copy(
                            recompiled_file,
                            f"red-team-success/{recompile_no_path}{optimization[opt]}{success_count_2}.c",
                        )
                        success_count_2 += 1
                    elif opt == 2:
                        assemble(
                            recompiled_file, recompile_no_path, func0, optimization[opt]
                        )
                        shutil.copy(
                            original_file,
                            f"red-team-success/{origin_no_path}{optimization[opt]}{success_count_3}.c",
                        )
                        shutil.copy(
                            recompiled_file,
                            f"red-team-success/{recompile_no_path}{optimization[opt]}{success_count_3}.c",
                        )
                        success_count_3 += 1

                    else:
                        pass
            except:
                if opt == 0:

                    shutil.copy(
                        original_file,
                        f"red-team-failures/{origin_no_path}{optimization[opt]}{garbage_file_count_1}.c",
                    )
                    shutil.copy(
                        recompiled_file,
                        f"red-team-failures/{recompile_no_path}{optimization[opt]}{garbage_file_count_1}.c",
                    )
                    garbage_file_count_1 += 1
                elif opt == 1:

                    shutil.copy(
                        original_file,
                        f"red-team-failures/{origin_no_path}{optimization[opt]}{garbage_file_count_2}.c",
                    )
                    shutil.copy(
                        recompiled_file,
                        f"red-team-failures/{recompile_no_path}{optimization[opt]}{garbage_file_count_2}.c",
                    )
                    garbage_file_count_2 += 1
                elif opt == 2:

                    shutil.copy(
                        original_file,
                        f"red-team-failures/{origin_no_path}{optimization[opt]}{garbage_file_count_3}.c",
                    )
                    shutil.copy(
                        recompiled_file,
                        f"red-team-failures/{recompile_no_path}{optimization[opt]}{garbage_file_count_3}.c",
                    )
                    garbage_file_count_3 += 1

    # TODO recompiled_file

    return 0


def assemble(name, name_no_path, function_name, optimization):
    asm_file_name = f"{name_no_path}.asm"
    s_file_name = f"{name_no_path}.s"
    obj_file_name = f"{name_no_path}.o"
    # os.system(f"gcc -c {file_name} -o {obj_file_name} && objdump -d {obj_file_name} > {asm_file_name}")
    # os.system(f"gcc -c {file_name} -o {obj_file_name}")

    compile_command = f"gcc -c {name} -o {obj_file_name} -{optimization} -lm"  # compile the code with GCC on Linux
    subprocess.run(compile_command, shell=True, check=True)
    compile_command = f"objdump -d {obj_file_name}> {s_file_name}"  # disassemble the binary file into assembly instructions
    subprocess.run(compile_command, shell=True, check=True)

    input_asm = ""
    with open(s_file_name) as f:  # asm file
        asm = f.read()
        if (
            "<" + function_name + ">:" not in asm
        ):  # IMPORTANT replace func0 with the function name
            raise ValueError("compile fails")
        asm = (
            "<"
            + function_name
            + ">:"
            + asm.split("<" + function_name + ">:")[-1].split("\n\n")[0]
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
    with open(asm_file_name, "w", encoding="utf-8") as f:
        f.write(input_asm_prompt)

    print(f"Assembly file generated: {asm_file_name}")

    return asm_file_name


def load_model():
    model_path = MODEL_PATH
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,  # Helps with large models
        device_map="auto",  # Automatically maps layers to available devices
        trust_remote_code=True,  # Trusts remote code, necessary for FlashAttention
    )
    return tokenizer, model


def decompiler(file_name, tokenizer, model):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    accelerator = Accelerator()

    # Load the tokenizer and model

    model = accelerator.prepare(model)

    # Read the assembly function
    # asm_file_path = f"{file_name}_{OPT_LEVELS[0]}.asm"
    # with open(asm_file_path, 'r') as f:

    # read the assembly function
    with open(file_name, "r") as f:
        asm_function = f.read()

    # Allocate the inputs tensor on the accelerator device
    inputs = tokenizer(asm_function, return_tensors="pt").to(accelerator.device)

    with torch.no_grad():
        # Use FlashAttention during generation
        outputs = model.generate(
            **inputs, max_new_tokens=2048
        )  # Max length to 4096, max new tokens should be below the range

    # Decode the generated output
    decompiled_function = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]) : -1])

    with torch.no_grad():
        torch.cuda.empty_cache()
    gc.collect()
    # Read the original C file
    # with open(file_name + '.c', 'r') as f:
    # original_function = f.read()

    # Print the original and decompiled functions
    # print(f"Original function:\n{original_function}")

    # Ask the user if they want to save the decompiled function to a file

    return decompiled_function


def save_failure():
    # TODO
    return 0


main()
