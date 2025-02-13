# Craig Kimball
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from accelerate import Accelerator

import numpy as np
import faiss

import os
import subprocess

import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")

# the dataset
index = faiss.IndexFlatL2(768)


# write assmebly embeddings to faiss index file
def save_index(index_l):
    faiss.write_index(index_l, "assemblyIndex.index")


# load a faiss index file from local
def read_index():
    return faiss.read_index("assemblyIndex.index")


# Create emedding for snippet
def embed_code(code_snippet):
    input = tokenizer(
        code_snippet, return_tensors="pt", truncation=True, max_length=128
    )
    outputs = model(**input)

    return outputs.last_hidden_state.mean(dim=1).detach().numpy().squeeze()


# Generate assmembly embeddings
def generate_embeddings():

    assembly_code_list = []

    # for file in os.listdir("asm_C/"):
    for file in asm:
        if file.endswith(".s"):
            path = os.path.join("asm_C/", file)

            with open(path, "r") as f:
                assembly_code_list.append(f.read())

    # print("\n\n\n\n\n".join(assembly_code_list))

    for assembly_code in assembly_code_list:
        embedding = embed_code(assembly_code)
        print(f"Embedding shape: {embedding.shape}")
        # assembly_embeddings.append(embedding)

    assembly_embeddings = np.array(
        [embed_code(assembly_code) for assembly_code in assembly_code_list]
    )
    index.add(assembly_embeddings)


# search index for closest c code based on new embedding vs datastore
def make_search(input_code):

    # TODO store c code outputs and read with index or generate when creating index
    c_code_outputs = []
    # for file in os.listdir("asm_C/"):
    for file in c:
        if file.endswith(".c"):
            path = os.path.join("asm_C/", file)

            with open(path, "r") as f:
                c_code_outputs.append(f.read())

    query_embedding = embed_code(input_code).reshape(1, -1)
    k = 1
    distance, indices = index.search(query_embedding, k)

    print("Index: ", indices)

    nearest = [c_code_outputs[i] for i in indices[0]]
    return nearest


# Operation

# generate_embeddings()
# save_index(index)

# index = read_index()

asm = [
    "factorial.s",
    "fibonacci.s",
    "floating_mult.s",
    "hello_world.s",
    "natural_sum_recursion.s",
    "num_swap.s",
    "powers_loop.s",
    "quadratic_roots.s",
    "quotient_remainder.s",
    "recursive_factorial.s",
]

c = [
    "factorial.c",
    "fibonacci.c",
    "floating_mult.c",
    "hello_world.c",
    "natural_sum_recursion.c",
    "num_swap.c",
    "powers_loop.c",
    "quadratic_roots.c",
    "quotient_remainder.c",
    "recursive_factorial.c",
]

generate_embeddings()

asm_file_path = "asm_C/powers_loop.s"

c_code = make_search(open(asm_file_path).read())

print("\n".join(c_code))

print("done")


# prepping LLM4D input:

# input_prompt = "; C Code Reference:\n; ________________ \n"
# input_prompt += "\n".join("\n".join(f"; {line}" for line in block.splitlines()) for block in c_code)
# input_prompt += "\n\n"

# input_prompt += open(asm_file_path).read()

# with open("Prompt.txt", "w") as prompt:
#    prompt.write(input_prompt)

# os.system("python ../LLM4Decompile_content/llm_decompiler.py -d -f Prompt.txt -t main -m 22b")


# prepping DeepSeek Coder

# input("Waiting")

deepseek = "deepseek-ai/deepseek-coder-6.7b-base"
tokenizer_chat = AutoTokenizer.from_pretrained(deepseek, trust_remote_code=True)
model_chat = AutoModelForCausalLM.from_pretrained(
    deepseek, torch_dtype=torch.bfloat16, trust_remote_code=True
).cuda()


input_text = "#Assembly Code: \n"
input_text += open(asm_file_path).read()
input_text += "\n"
input_text += "\n\n# C Code: \n"
input_text += "\n".join(c_code)
input_text += "\n"

input_text += "# Decompile the above Assembly code"

input_text = "#Write a quick sort algorithm"

# accelerator = Accelerator()
# model_chat = accelerator.prepare(model_chat)


inputs = tokenizer_chat(input_text, return_tensors="pt").to(model_chat.device)

outputs = model_chat.generate(**inputs, max_length=2048)  # max_length=128


print(tokenizer_chat.decode(outputs[0], skip_special_tokens=True))
