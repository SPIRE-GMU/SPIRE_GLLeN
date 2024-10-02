from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn

OPT = ["O0", "O1", "O2", "O3"]
fileName = 'test'
device = torch.device("cuda:0")
tensor = torch.tensor([0,1],device=device)
model_path = '/home/spire2/LLM4Decompile/llm4decompile-22b-v2' # V1.5 Model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path,device_map = 'auto', torch_dtype=torch.bfloat16).cuda()
model = model.to(device)
model = nn.DataParallel(model)

with open(fileName +'_' + OPT[0] +'.asm','r') as f:#optimization level O0
    asm_func = f.read()
inputs = tokenizer(asm_func, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=2048)### max length to 4096, max new tokens should be below the range
c_func_decompile = tokenizer.decode(outputs[0][len(inputs[0]):-1])

with open(fileName +'.c','r') as f:#original file
    func = f.read()

print(f'original function:\n{func}')# Note we only decompile one function, where the original file may contain multiple functions
print(f'decompiled function:\n{c_func_decompile}')
