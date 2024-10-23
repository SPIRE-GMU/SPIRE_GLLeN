import os
import subprocess
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
import argparse


"""
Version: 1.0

TODO Adjust input control flow to accept .o object files and parse to assembly
TODO Adjust directories to dynamical locate models
TODO Add multi function support for decompile
TODO Add opp codes for future use 

"""

TITLE_CARD = r"""
    __    __    __  _____ __ 
   / /   / /   /  |/  / // /  ___                                  _ __
  / /   / /   / /|_/ / // /_/ __ \___  _________  ____ ___  ____  (_) /__ 
 / /___/ /___/ /  / /__  __/ / / / _ \/ ___/ __ \/ __ `__ \/ __ \/ / / _ \
/_____/_____/_/  /_/  / / / /_/ /  __/ /__/ /_/ / / / / / / /_/ / / /  __/
                     /_/ /_____/\___/\___/\____/_/ /_/ / / .___/_/_/\___/  
                                                        /_/
"""


OPT_LEVELS = ["O0", "O1", "O2", "O3"]

# TODO resolve hardcoded paths to models
MODEL_PATHS = {
    "6.7b": "/home/spire2/LLM4Decompile/llm4decompile-6.7b-v2",
    "9b": "/home/spire2/LLM4Decompile/llm4decompile-9b-v2",
    "22b": "/home/spire2/LLM4Decompile/llm4decompile-22b-v2",
}

func0 = ""

"""
Load tokenizer and model with FlashAttention

@param model_size: The siez of the model to load
@return: a tuple containing the loaded tokenizer and model
"""


def load_model(model_size):
    model_path = MODEL_PATHS[model_size]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,  # Helps with large models
        device_map="auto",  # Automatically maps layers to available devices
        trust_remote_code=True,  # Trusts remote code, necessary for FlashAttention
    )
    return tokenizer, model


"""
Assembled the C code into an executable file and asm file using GCC

@param file_name: assumes .c ending string, the path to file
"""


def assemble(file_name, function_name):
    """Assemble the C code"""
    # Get the file name without the extension
    if file_name.endswith(".c"):
        file_name_without_ext = file_name[:-2]
    else:
        file_name_without_ext = file_name

    # Compile the C code to assembly and object files
    asm_file_name = f"{file_name_without_ext}.asm"
    s_file_name = f"{file_name_without_ext}.s"
    obj_file_name = f"{file_name_without_ext}.o"
    # os.system(f"gcc -c {file_name} -o {obj_file_name} && objdump -d {obj_file_name} > {asm_file_name}")
    # os.system(f"gcc -c {file_name} -o {obj_file_name}")

    compile_command = (
        f"gcc -o {obj_file_name} {file_name} -lm"  # compile the code with GCC on Linux
    )
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
    print(f"Object file generated: {obj_file_name}")

    return 0


"""
Runs the LLM4Decompile process

@param file_name: The name of the file to decompile.
@param model_size: The size of the model to use for decompilation.
@param func_name: The name of the function to decompile.
@return: An integer indicating the success of the decompilation process.
"""


def run(file_name, model_size, func_name):
    # Initialize the accelerator
    accelerator = Accelerator()

    # Load the tokenizer and model
    tokenizer, model = load_model(model_size)
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

    # Read the original C file
    # with open(file_name + '.c', 'r') as f:
    # original_function = f.read()

    # Print the original and decompiled functions
    # print(f"Original function:\n{original_function}")
    print(f"Decompiled function:\n{decompiled_function}")

    # Ask the user if they want to save the decompiled function to a file
    save_to_file = input(
        "Do you want to save the decompiled function to a file? (y/n): "
    )
    if save_to_file.lower() == "y":
        file_name = input("Enter a file name: ")
        with open(file_name, "w") as f:
            f.write(decompiled_function)
        print(f"Decompiled function saved to {file_name}")
    else:
        print("Decompiled function not saved to a file.")

    return 0

    return 0


def main():
    parser = argparse.ArgumentParser(description="Decompile assembly code")
    parser.add_argument("-c", "--compile", action="store_true", help="Compile the code")
    parser.add_argument(
        "-d", "--decompile", action="store_true", help="Decompile the assembly code"
    )
    parser.add_argument("-f", "--file", type=str, help="Input file path")
    parser.add_argument("-t", "--function", type=str, help="Function name to decompile")
    parser.add_argument("-m", "--model", type=str, help="Model size (6.7b, 9b, 22b)")
    args = parser.parse_args()

    if args.compile:
        assemble(args.file, args.model, args.function)
    elif args.decompile:
        if args.file and args.function and args.model:
            if args.model not in MODEL_PATHS:
                raise ValueError(
                    "Invalid model option. Choose from '6.7b', '9b', or '22b'."
                )
            run(args.file, args.model, args.function)
        else:
            print("Error: Please provide the file name, function name, and model size.")
    else:
        # Clear the terminal screen
        os.system("clear")

        # Display the ASCII art
        print(TITLE_CARD)

        # Get a list of files with .c or .o extensions in the current working directory
        current_dir = os.path.dirname(__file__)
        files = [
            f for f in os.listdir(current_dir) if f.endswith(".c") or f.endswith(".asm")
        ]

        # Prompt the user for the file name and display the list of files
        print("Select the file name of the file:")
        # print("Available files:")
        for i, file in enumerate(files, start=1):
            print(f"{i}. {file}")

        # Get user input as a string
        file_input = input()

        # Check if the input is a digit
        if file_input.isdigit():
            file_number = int(file_input)
            if file_number < 1 or file_number > len(files):
                raise ValueError("Invalid file number.")
            file_name = files[file_number - 1]
        else:
            # Check if the input is a file name
            if file_input in files:
                file_name = file_input
            else:
                raise ValueError("Invalid file name.")

        if file_name.endswith(".c"):

            # Get user input for the function name
            print("Enter the function name to decompile: [main]")
            func0 = input()

            if assemble(file_name, func0) == 0:
                print("Compilation successful.")

            return 0

        else:
            # Get user input for the model size

            # TODO resolve hardcoded model sizes, gather from installation location
            print("Select the model size: ")
            print("1. 6.7b")
            print("2. 9b")
            print("3. 22b")
            model_input = input()

            # Check if the input is a digit
            if model_input.isdigit():
                model_numbers = {"1": "6.7b", "2": "9b", "3": "22b"}
                if model_input in model_numbers:
                    model_size = model_numbers[model_input]
                else:
                    raise ValueError(
                        "Invalid model option. Choose from '1', '2', or '3'."
                    )
            else:
                # Check if the input is a model size
                model_sizes = ["6.7b", "9b", "22b"]
                if model_input in model_sizes:
                    model_size = model_input
                else:
                    raise ValueError(
                        "Invalid model option. Choose from '6.7b', '9b', or '22b'."
                    )

            # Get user input for the function name
            print("Enter the function name to decompile: [main]")
            func0 = input()

            # Run the decompilation process
            return run(file_name, model_size, func0)


if __name__ == "__main__":
    while True:
        main()
        if input("Do you want to compile/decompile more files? (y/n)").lower() != "y":
            break
