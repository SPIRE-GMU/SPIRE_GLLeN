from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True)

# Load the model in bf16 and move it to the GPU
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-coder-1.3b-base",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16  # Specify bf16 data type
).cuda()

# Prepare input text and move to the model's device
input_text = "#write a quick sort algorithm"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

# Generate and decode the output
outputs = model.generate(**inputs, max_length=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

