import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
OPT = ["O0", "O1", "O2", "O3"]

# Initialize the accelerator
accelerator = Accelerator()

# Define model paths
model_paths = {
    '6.7b': '/home/spire2/LLM4Decompile/llm4decompile-6.7b-v2',
    '9b': '/home/spire2/LLM4Decompile/llm4decompile-9b-v2',
    '22b': '/home/spire2/LLM4Decompile/llm4decompile-22b-v2'
}

# Example usage: define the file name and model option
user_file = input("Enter the file name to the C file (without extension): ")  # e.g., 'samples/sample'
user_file = os.path.join('/home/spire2/TestWork', user_file)  # Construct full path

user_model = input("Choose a model (6.7b, 9b, 22b): ")  # e.g., '6.7b'
model_path = model_paths.get(user_model)

if model_path is None:
    raise ValueError("Invalid model option. Choose from '6.7b', '9b', or '22b'.")

# Load the tokenizer and model with FlashAttention
tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,  # Helps with large models
    device_map="auto",  # Automatically maps layers to available devices
    trust_remote_code=True  # Trusts remote code, necessary for FlashAttention
)

# Move the model to the accelerator (multiple GPUs)
model = accelerator.prepare(model)

# Open and read the assembly function
OPT = ["O0", "O1", "O2", "O3"]
asm_file_path = f"{user_file}_{OPT[0]}.asm"  # Assuming OPT[0] is defined earlier in the code
with open(asm_file_path, 'r') as f:  # optimization level O0
    asm_func = f.read()

# Allocate the inputs tensor on the accelerator device
inputs = tokenizer(asm_func, return_tensors="pt").to(accelerator.device)

with torch.no_grad():
    # Use FlashAttention during generation
    outputs = model.generate(**inputs, max_new_tokens=2048)  # Max length to 4096, max new tokens should be below the range

# Decode the generated output
c_func_decompile = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):-1])

# Open and read the original C file
with open(user_file + '.c', 'r') as f:  # original file
    func = f.read()

# Print the original and decompiled functions
print(f'Original function:\n{func}')  # Note we only decompile one function, where the original file may contain multiple functions
print(f'Decompiled function:\n{c_func_decompile}')
