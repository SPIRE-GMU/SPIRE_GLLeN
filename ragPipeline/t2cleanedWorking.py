# Author: Craig Kimball
import os
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")

# FAISS index for embedding storage
index = faiss.IndexFlatL2(768)

# Save FAISS index to a file
def save_index(index_l, file_path="assemblyIndex.index"):
    faiss.write_index(index_l, file_path)

# Load FAISS index from a file
def load_index(file_path="assemblyIndex.index"):
    return faiss.read_index(file_path)

# Create embedding for a code snippet
def embed_code(code_snippet):
    inputs = tokenizer(code_snippet, return_tensors="pt", truncation=True, max_length=128)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy().squeeze()

# Generate embeddings for assembly code files
def generate_embeddings(assembly_files, index):
    assembly_code_list = []

    for file in assembly_files:
        file_path = os.path.join("asm_C/", file)
        with open(file_path, 'r') as f:
            assembly_code_list.append(f.read())

    # Embed all assembly code snippets
    assembly_embeddings = np.array([embed_code(code) for code in assembly_code_list])
    index.add(assembly_embeddings)

# Search for the closest C code match for a given assembly code
def make_search(input_code, index, c_code_files):
    c_code_outputs = []

    for file in c_code_files:
        file_path = os.path.join("asm_C/", file)
        with open(file_path, 'r') as f:
            c_code_outputs.append(f.read())

    query_embedding = embed_code(input_code).reshape(1, -1)
    k = 1  # Number of nearest neighbors to retrieve
    distances, indices = index.search(query_embedding, k)

    nearest_c_code = [c_code_outputs[i] for i in indices[0]]
    return nearest_c_code

# Define file lists
asm_files = [
    "factorial.s", "fibonacci.s", "floating_mult.s", "hello_world.s",
    "natural_sum_recursion.s", "num_swap.s", "powers_loop.s",
    "quadratic_roots.s", "quotient_remainder.s", "recursive_factorial.s"
]

c_files = [
    "factorial.c", "fibonacci.c", "floating_mult.c", "hello_world.c",
    "natural_sum_recursion.c", "num_swap.c", "powers_loop.c",
    "quadratic_roots.c", "quotient_remainder.c", "recursive_factorial.c"
]

# Generate and save embeddings
generate_embeddings(asm_files, index)
save_index(index)

# Load the index and search for C code
index = load_index()
asm_file_path = "asm_C/powers_loop.s"

with open(asm_file_path, 'r') as asm_file:
    input_asm_code = asm_file.read()

matched_c_code = make_search(input_asm_code, index, c_files)

print("\n".join(matched_c_code))
print("Search completed.")

# Prepare input for a DeepSeek Coder or other downstream tasks
deepseek_model_path = "deepseek-ai/deepseek-coder-6.7b-base"
tokenizer_chat = AutoTokenizer.from_pretrained(deepseek_model_path, trust_remote_code=True)
model_chat = AutoModelForCausalLM.from_pretrained(
    deepseek_model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).cuda()

input_text = (
    "# Assembly Code:\n" + input_asm_code +
    "\n\n# C Code:\n" + "\n".join(matched_c_code) +
    "\n\n# Decompile the above Assembly code"
)

# Tokenize and generate output
inputs = tokenizer_chat(input_text, return_tensors="pt").to(model_chat.device)
outputs = model_chat.generate(**inputs, max_length=2048)

print(tokenizer_chat.decode(outputs[0], skip_special_tokens=True))
