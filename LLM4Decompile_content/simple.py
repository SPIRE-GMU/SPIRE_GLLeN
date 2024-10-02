import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set model path
model_path = '/home/spire2/LLM4Decompile/llm4decompile-22b-v2'

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

# Load the model in mixed precision
base_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, local_files_only=True)

# Inspect the model architecture to find the correct attribute for layers
print(base_model)  # This will print the model's structure
