import subprocess
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from accelerate import Accelerator

# Set environment variable to suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set the environment variable to limit fragment size
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Clear CUDA cache at the start
torch.cuda.empty_cache()

# Set PyTorch memory management settings
torch.cuda.set_per_process_memory_fraction(0.9)


def preprocess_and_decompile(file_name, model_option):
    # Set optimization levels
    OPT = ["O0", "O1", "O2", "O3"]
    func0 = "main"

    # Check the selected model path
    model_paths = {
        "6.7b": "/home/spire2/LLM4Decompile/llm4decompile-6.7b-v2",
        "9b": "/home/spire2/LLM4Decompile/llm4decompile-9b-v2",
        "22b": "/home/spire2/LLM4Decompile/llm4decompile-22b-v2",
    }
    model_path = model_paths.get(model_option)
    if model_path is None:
        raise ValueError("Invalid model option. Choose from '6.7b', '9b', or '22b'.")

    # Accelerator for multi-GPU and mixed precision
    accelerator = Accelerator(mixed_precision="fp16", device_placement=True)

    # Preprocessing: Compile and disassemble
    for opt_state in OPT:
        output_file = f"{file_name}_{opt_state}"
        input_file = f"{file_name}.c"

        try:
            # Compile the code with GCC on Linux
            compile_command = f"gcc -o {output_file}.o {input_file} -{opt_state} -lm"
            subprocess.run(compile_command, shell=True, check=True)

            # Disassemble the binary file into assembly instructions
            compile_command = f"objdump -d {output_file}.o > {output_file}.s"
            subprocess.run(compile_command, shell=True, check=True)

            # Read and clean the assembly instructions
            with open(f"{output_file}.s") as f:
                asm = f.read()
                if f"<{func0}>:" not in asm:
                    raise ValueError("Compilation fails; function not found.")

                asm = f"<{func0}>:" + asm.split(f"<{func0}>:")[-1].split("\n\n")[0]

                asm_clean = ""
                for line in asm.split("\n"):
                    if len(line.split("\t")) < 3 and "00" in line:
                        continue
                    idx = min(len(line.split("\t")) - 1, 2)
                    tmp_asm = "\t".join(
                        line.split("\t")[idx:]
                    )  # Remove the binary code
                    tmp_asm = tmp_asm.split("#")[0].strip()  # Remove comments
                    asm_clean += tmp_asm + "\n"

            input_asm = asm_clean.strip()
            input_asm_prompt = f"# This is the assembly code:\n{input_asm.strip()}\n# What is the source code?\n"

            with open(f"{output_file}.asm", "w", encoding="utf-8") as f:
                f.write(input_asm_prompt)

            # Decompilation: Use LLM4Decompile to translate assembly instructions into C
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.float16
            )

            # Enable gradient checkpointing to save memory
            model.gradient_checkpointing_enable()

            # Monitor memory before preparing the model
            print(
                f"Allocated memory before loading model: {torch.cuda.memory_allocated()} bytes"
            )
            print(
                f"Cached memory before loading model: {torch.cuda.memory_reserved()} bytes"
            )

            print(
                f"Model parameters before prepare: {sum(p.numel() for p in model.parameters())}"
            )

            # Prepare model for distributed training
            model, inputs = accelerator.prepare(
                model,
                tokenizer(
                    input_asm_prompt, return_tensors="pt", truncation=True, padding=True
                ),
            )

            print(
                f"Model parameters after prepare: {sum(p.numel() for p in model.parameters())}"
            )
            print(
                f"Allocated memory after loading model: {torch.cuda.memory_allocated()} bytes"
            )
            print(
                f"Cached memory after loading model: {torch.cuda.memory_reserved()} bytes"
            )

            with open(f"{output_file}.asm", "r") as f:  # Read the assembly
                asm_func = f.read()

            # Tokenize the assembly function, only necessary parts
            inputs = tokenizer(
                asm_func, return_tensors="pt", truncation=True, padding=True
            )

            # Move inputs to the same device as the model
            inputs = inputs.to(
                accelerator.device
            )  # Ensure inputs are on the same device (GPU)

            # Generate the output
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=1024)

            # Clear CUDA cache after generation
            torch.cuda.empty_cache()

            if outputs.size(1) > len(inputs["input_ids"][0]):
                c_func_decompile = tokenizer.decode(
                    outputs[0][len(inputs["input_ids"][0]) : -1]
                )
            else:
                c_func_decompile = ""

            with open(f"{file_name}.c", "r") as f:  # Original file
                func = f.read()

            # Print results
            print(f"Original function:\n{func}")  # Original function
            print(f"Decompiled function:\n{c_func_decompile}")  # Decompiled function

        except subprocess.CalledProcessError as e:
            print(f"Error occurred during processing: {e}")


if __name__ == "__main__":
    # Example usage: define the file name and model option
    user_file = input(
        "Enter the file name to the C file (without extension): "
    )  # e.g., 'samples/sample'
    user_file = os.path.join("/home/spire2/TestWork", user_file)  # Construct full path
    user_model = input("Choose a model (6.7b, 9b, 22b): ")  # e.g., '6.7b'

    preprocess_and_decompile(user_file, user_model)
