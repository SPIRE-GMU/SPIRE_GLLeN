from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

OPT = ["O0", "O1", "O2", "O3"]
fileName = 'test'
model_path = '/home/spire2/LLM4Decompile/llm4decompile-22b-v2'

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load model
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)

# Wrap model in DataParallel
device_ids = [0, 1]
model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()

# Load assembly function
with open(fileName + '_' + OPT[0] + '.asm', 'r') as f:  # optimization level O0
    asm_func = f.read()

# Tokenize input and send it to the GPU
inputs = tokenizer(asm_func, return_tensors="pt").to('cuda:0')  # Ensure the inputs are on the same device as DataParallel's default

# Generate output without gradient computation
with torch.no_grad():
    outputs = model.module.generate(**inputs, max_new_tokens=2048)  # Use model.module to access generate method

# Decode decompiled function
c_func_decompile = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):-1])

# Load original C function
with open(fileName + '.c', 'r') as f:
    func = f.read()

# Print original and decompiled function
print(f'Original function:\n{func}')  # Note: we only decompile one function
print(f'Decompiled function:\n{c_func_decompile}')
