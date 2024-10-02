from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
import torch
import os


def main():
    # Initialize accelerator with mixed precision for memory savings
    accelerator = Accelerator(
        mixed_precision="fp16"
    )  # Use FP16 for memory optimization

    model_path = "/home/spire2/LLM4Decompile/llm4decompile-9b-v2"  # V1.5 Model
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load the model with mixed precision (bfloat16 for faster computation and less memory)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)

    # Prepare model and tokenizer using accelerator
    model = accelerator.prepare(model)

    # Read input file
    fileName = "your_file_name"  # Replace with your filename
    with open(fileName + "_" + "OPT[0]" + ".asm", "r") as f:  # Optimization level O0
        asm_func = f.read()

    # Tokenize the input
    inputs = tokenizer(asm_func, return_tensors="pt", padding=True, truncation=True)

    # Prepare inputs with accelerator
    inputs = accelerator.prepare(inputs)

    # Clear any cached memory before running inference
    torch.cuda.empty_cache()

    # Generate output using the model, limiting tokens to avoid memory overload
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1024)  # Reduced token size

    # Decode the output (cut off the input length from the generated sequence)
    c_func_decompile = tokenizer.decode(
        outputs[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True
    )

    # Read the original file for comparison
    with open(fileName + ".c", "r") as f:  # Original file
        func = f.read()

    # Print the original and decompiled functions
    print(f"Original function:\n{func}")  # Only decompile one function
    print(f"Decompiled function:\n{c_func_decompile}")


if __name__ == "__main__":
    # Set environment variable to avoid memory fragmentation issues
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    # Run the main function
    main()
