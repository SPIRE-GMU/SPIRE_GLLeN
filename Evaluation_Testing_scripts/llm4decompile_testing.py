"""
This file iterates through the exebench
dataset and tests if LLM4Decompile is able to 
decompile a recombilable C file
"""

from datasets import load_dataset
import gc
import os
import shutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
import subprocess
import re

MODEL_PATH = "LLM4Decompile/llm4decompile-22b-v2"
c_file_dir = "c_files/"
success_file_dir = "llm4_success_files/"
garbage_dir = "llm4_garbage/"
fail_dir = "llm4_fail_files/"
c_temp_dir = "llm4_temp/"


def main():
    """
    The main function controling all other functions
    """
    tokenizer, model = load_model()
    count = {"Success": 0, "Fail": 0}

    for file in os.listdir(c_file_dir):
        if file.endswith(".c"):
            c_path = os.path.join(c_file_dir, file)
            filename = os.path.splitext(os.path.basename(c_path))[0]
            name_list = re.split("_(\d+)", filename)
            func_name = name_list[0]
            count = decompile_counter(
                c_path, count, model, tokenizer, filename, func_name
            )

    return count

    # 1) Load dataset split. In this case, synthetic test split


def decompile_counter(c_path, count, model, tokenizer, filename, func_name):

    try:
        asm_file = assemble(c_path, garbage_dir, filename, func_name)
        decompiled_func = decompiler(asm_file, tokenizer, model)
        temp_path = c_temp_dir + filename + ".c"
        with open(temp_path, "w") as d:
            d.write(decompiled_func)
            d.close()

        currently_a_success = True

        # Clear the terminal screen

        # Display the ASCII art

        # Get a list of files with .c or .o extensions in the current working directory

    except:
        print("This file sucks becuse exebench sucks")
        currently_a_success = False

    if currently_a_success:
        count = recompile(temp_path, success_file_dir, fail_dir, count, garbage_dir)

    # TODO recompiled_file
    print(count)
    return count


def recompile(file_path, success_path, fail_path, count, garbage_folder):
    filename = os.path.splitext(os.path.basename(file_path))[0]

    try:
        gcc_asm_command = [
            "gcc",
            "-S",
            file_path,
            "-o",
            os.path.join(garbage_folder, f"{filename}.s"),
        ]
        subprocess.run(gcc_asm_command, check=True, capture_output=True)
        shutil.copy(file_path, success_path)
        count["Success"] = count["Success"] + 1

        return count
    except:
        count["Fail"] = count["Fail"] + 1
        shutil.copy(file_path, fail_path)
        return count


def assemble(file_path, garbage_path, filename, func_name):

    obj_file_name = garbage_path + filename + ".o"
    s_file_name = garbage_path + filename + ".s"
    asm_file_name = garbage_path + filename + ".asm"

    compile_command = f"gcc -c -o {obj_file_name} {file_path} -lm"  # compile the code with GCC on Linux
    subprocess.run(compile_command, shell=True, check=True)
    compile_command = f"objdump -d {obj_file_name}> {s_file_name}"  # disassemble the binary file into assembly instructions
    subprocess.run(compile_command, shell=True, check=True)
    input_asm = ""
    with open(s_file_name) as f:  # asm file
        asm = f.read()
        if (
            "<" + func_name + ">:" not in asm
        ):  # IMPORTANT replace func0 with the function name
            raise ValueError("compile fails")
        asm = (
            "<"
            + func_name
            + ">:"
            + asm.split("<" + func_name + ">:")[-1].split("\n\n")[0]
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
    print(f"Object file generated: {obj_file_name}")
    return asm_file_name

    # with open(asm_file) as f:  # asm file
    #     asm = f.read()
    #     if (
    #         "<" + function_name + ">:" not in asm
    #     ):  # IMPORTANT replace func0 with the function name
    #         raise ValueError("compile fails")
    #     asm = (
    #         "<"
    #         + function_name
    #         + ">:"
    #         + asm.split("<" + function_name + ">:")[-1].split("\n\n")[0]
    #     )  # IMPORTANT replace func0 with the function name
    #     asm_clean = ""
    #     asm_sp = asm.split("\n")
    #     for tmp in asm_sp:
    #         if len(tmp.split("\t")) < 3 and "00" in tmp:
    #             continue
    #         idx = min(len(tmp.split("\t")) - 1, 2)
    #         tmp_asm = "\t".join(tmp.split("\t")[idx:])  # remove the binary code
    #         tmp_asm = tmp_asm.split("#")[0].strip()  # remove the comments
    #         asm_clean += tmp_asm + "\n"
    # input_asm = asm_clean.strip()
    # before = f"# This is the assembly code:\n"  # prompt
    # after = "\n# What is the source code?\n"  # prompt
    # input_asm_prompt = before + input_asm.strip() + after
    # with open(asm_file_name, "w", encoding="utf-8") as f:
    #     f.write(input_asm_prompt)

    # print(f"Assembly file generated: {asm_file_name}")

    # return asm_file_name


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
