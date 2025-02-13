"""
Craig Kimball

This file handles the RAG pipeline using CFG's in Neo4J
Version 1 uses the following pipeline
1) Assmebly to C Code using specified model (DeepSeek Coder)
2) C Code to CFG
3) CFG KNN search in NEO4j Database
4) Original Assmebly and CFG are fed to specified model for final C code decompile

5) Similarity Matching with SMT to determine effectiveness

"""

import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
import torch

import os
import subprocess
from dotenv import load_dotenv

from neo4j import GraphDatabase
import numpy as np

# from sklearn.neighbors import NearestNeighbors


def asmToC(asmCode):
    input_text = "Decompile the following assembly code to C code:"

    with open(asmCode, "r") as file:
        asm_function = file.read()

        input_asm_code = asm_function

    input = (
        "Assembly Code:\n" + input_asm_code + "\n\n Decompile the above Assembly Code"
    )

    # print(input)
    # input = "Write a simple python buble sort method and return only the code"
    results = querryModel(input)
    return results[2:]


def cToCfg(c_file_name):

    print("temp_c/" + c_file_name.split(".")[0])

    gcc_cfg_command = [
        "gcc",  # Adjusted to use gcc 11.4.0 for ubuntu
        "-fdump-tree-all-graph",
        "-c",  # Compile only, do not link (suitable for files without main)
        c_file_name,
        "-o",
        c_file_name.split(".")[0],
    ]

    result_cfg = subprocess.run(
        gcc_cfg_command, check=True, capture_output=True, text=True
    )

    return c_file_name.split(".")[0] + ".out"


def querryModel(input_text):
    """Prepare input for a DeepSeek Coder or other downstream tasks"""
    # Only DeepSeek Handled Here
    deepseek_model_path = "deepseek-ai/deepseek-coder-6.7b-base"
    # deepseek_model_path = "deepseek-ai/deepseek-coder-1.3b-base"
    # deepseek_model_path = "deepseek-ai/deepseek-coder-6.7b-instruct"
    # deepseek_model_path = "deepseek-ai/deepseek-coder-1.3b-instruct"

    # deepseek_model_path = "deepseek-ai/deepseek-r1-distill-qwen-14b"
    # deepseek_model_path = "deepseek-ai/deepseek-r1-distill-llama-8b"

    tokenizer_chat = AutoTokenizer.from_pretrained(
        deepseek_model_path, trust_remote_code=True
    )
    model_chat = AutoModelForCausalLM.from_pretrained(
        deepseek_model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto",
    )

    accelerator = Accelerator()
    model_chat = accelerator.prepare(model_chat)

    # assemble files to format model can understand

    # input_text = "#write a single bubble sort algorith in python"

    # Tokenize and generate output
    inputs = tokenizer_chat(input_text, return_tensors="pt").to(accelerator.device)
    outputs = model_chat.generate(**inputs, max_length=3000)

    # print("\n\n\nResponse:\n\n\n")
    return tokenizer_chat.decode(
        outputs[0][len(inputs["input_ids"][0]) : -1], skip_special_tokens=True
    )


# Connect to Neo4j
class Neo4jConnection:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def get_cfgs(self):
        with self.driver.session() as session:
            result = session.run("MATCH (c:CFG) RETURN c.id AS id, c.vector AS vector")
            return [(record["id"], np.array(record["vector"])) for record in result]


""" Initialize neo4j variables """
load_dotenv()
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_URI = os.getenv("NEO4J_CONNECTION_URI")

# Replace with your Neo4j credentials
# neo4j_conn = Neo4jConnection(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Passed Arguments
# First argument is source assmembly code to decompile
source_asm = sys.argv[1]


# 1) Assmebly to C Code using specified model (DeepSeek Coder)
created_c = asmToC(source_asm)
print(created_c)

c_out_file_name = "temp_c/" + source_asm.split(".")[0] + "_out.c"

with open(c_out_file_name, "w") as file:
    file.write(created_c)


# 2) C Code to CFG
cfg_file_name = cToCfg(c_out_file_name)

# 3) CFG KNN search in NEO4j Database
for filename in os.listdir("temp_c"):
    # Check if the file ends with .cfg
    if filename.endswith(".cfg"):
        cfg_file_name = filename
print(cfg_file_name)

# 4) Original Assmebly and CFG are fed to specified model for final C code decompile

# 5) Similarity Matching with SMT to determine effectiveness
