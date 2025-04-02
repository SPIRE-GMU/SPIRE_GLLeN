"""
Craig Kimball with parts from Justin Rockwell

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
import r2pipe

import KNN_search

# from sklearn.neighbors import NearestNeighbors


def asmToC(asmCode):
    input_text = "Decompile the following assembly code to C code:"

    with open(asmCode, "r") as file:
        asm_function = file.read()

        input_asm_code = asm_function

    input = (
        "Assembly Code:\n" + input_asm_code + "\n\nDecompile the above Assembly Code and return only C code with no other text, explinations, or formatting"
    )

    input =  "Assembly Code:\n" + input_asm_code + "\n\nDecompile the above Assembly Code into C code formatted with the C code surrounded by ``` to indicate where the code is"

    print(input)
    # input = "Write a simple python buble sort method and return only the code"
    results = querryModel(input)
    print(results)
    # input("Test Did work?")
    # print("+++++")
    return results


def cToCfg(c_file_name):

    print("temp_c/" + c_file_name.split(".")[0])

    gcc_cfg_command = [
        "gcc",  # Adjusted to use gcc 11.4.0 for ubuntu
        "-fdump-tree-cfg-graph",
        "-c",  # Compile only, do not link (suitable for files without main)
        c_file_name,
        "-o",
        c_file_name.split(".")[0],
    ]

    result_cfg = subprocess.run(
        gcc_cfg_command, check=True, capture_output=True, text=True
    )

    return c_file_name.split(".")[0] + ".out"

def asmToCFG(dot_file, obj_file, func_name):
    #r2 takes input object file.

    # Step 3: Open radare2 (r2pipe) and analyze the object file
    print(f"[DEBUG] Opening radare2 on {obj_file} with function name = {func_name}")
    r2 = r2pipe.open(obj_file, flags=["-2"])  # quiet mode
    print("[DEBUG] Running 'aaa' (auto analyze all)...")
    r2.cmd("aaa")

    # Step 4: List functions (afl) for debugging
    print("[DEBUG] Output of 'afl':")
    afl_output = r2.cmd("afl")
    print(afl_output)

    # Step 5: Attempt to export the CFG in DOT format using the provided function name
    agfd_cmd = f"agfd @ {func_name}"
    print(f"[DEBUG] Running '{agfd_cmd}' ...")
    dot_data = r2.cmd(agfd_cmd)

    # Fallback: if no data, try prefixing with "sym."
    if not dot_data.strip():
        print("[WARN] DOT data is empty. Trying with 'sym.' prefix...")
        agfd_cmd = f"agfd @ sym.{func_name}"
        print(f"[DEBUG] Running '{agfd_cmd}' ...")
        dot_data = r2.cmd(agfd_cmd)

    r2.quit()

    if dot_data.strip() == "":
        print("[ERROR] DOT data is still empty. No CFG was generated for the function.")
    else:
        print(f"[DEBUG] DOT data length: {len(dot_data)} characters")

    # Step 6: Write the DOT data to file
    with open(dot_file, "w") as f:
        f.write(dot_data)
    print(f"[DEBUG] DOT CFG saved to '{dot_file}'")
    

def objToS(out_dir, object_file):
    assembly_file_path = os.path.join(out_dir, os.path.splitext(os.path.basename(object_file))[0] + '.s')
    
    # Use objdump to disassemble the object file
    with open(assembly_file_path, 'w') as f:
        subprocess.run(['objdump', '-d', object_file], 
                      check=True, stdout=f)

    # Parse the assembly file to extract function name
    function_name = None
    with open(assembly_file_path, 'r') as f:
        for line in f:
            # Look for lines containing function declarations
            if '<' in line and '>:' in line:
                # Extract the function name between < and >
                start = line.find('<') + 1
                end = line.find('>')
                function_name = line[start:end]
                break  # Stop after finding the first function
    
    return assembly_file_path, function_name

def loadModel():
    """Prepare input for a DeepSeek Coder or other downstream tasks"""
    # Only DeepSeek Handled Here
    # deepseek_model_path = "deepseek-ai/deepseek-coder-6.7b-base"
    # deepseek_model_path = "deepseek-ai/deepseek-coder-1.3b-base"
    # deepseek_model_path = "deepseek-ai/deepseek-coder-6.7b-instruct"
    # deepseek_model_path = "deepseek-ai/deepseek-coder-1.3b-instruct"

    deepseek_model_path = "deepseek-ai/deepseek-r1-distill-qwen-14b"
    # deepseek_model_path = "deepseek-ai/deepseek-r1-distill-llama-8b"

    global tokenizer_chat 
    tokenizer_chat = AutoTokenizer.from_pretrained(
        deepseek_model_path, trust_remote_code=True
    )
    global model_chat 
    model_chat = AutoModelForCausalLM.from_pretrained(
        deepseek_model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto",
    )

    global accelerator 
    accelerator = Accelerator()

    model_chat = accelerator.prepare(model_chat)

    # assemble files to format model can understand

    # input_text = "#write a single bubble sort algorith in python"

def querryModel(input_text):
    # Tokenize and generate output
    inputs = tokenizer_chat(input_text, return_tensors="pt").to(accelerator.device)
    outputs = model_chat.generate(**inputs, max_length=5000)

    # print("\n\n\nResponse:\n\n\n")
    return tokenizer_chat.decode(
        outputs[0][len(inputs["input_ids"][0]) : -1], skip_special_tokens=True
    )


# # Connect to Neo4j
# class Neo4jConnection:
#     def __init__(self, uri, user, password):
#         self.driver = GraphDatabase.driver(uri, auth=(user, password))

#     def close(self):
#         self.driver.close()

#     def get_cfgs(self):
#         with self.driver.session() as session:
#             result = session.run("MATCH (c:CFG) RETURN c.id AS id, c.vector AS vector")
#             return [(record["id"], np.array(record["vector"])) for record in result]


# """ Initialize neo4j variables """
# load_dotenv()
# NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
# NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
# NEO4J_URI = os.getenv("NEO4J_CONNECTION_URI")

# # Replace with your Neo4j credentials
# # neo4j_conn = Neo4jConnection(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)

# driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Passed Arguments
# First argument is source assmembly code to decompile

def main():

    # source_asm = sys.argv[1]
    source_obj = sys.argv[1]

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    # loadModel()

    # 1.A) Assmebly to C Code using specified model (DeepSeek Coder)
    # created_c = asmToC(source_asm)
    # created_c = created_c.split("```")[1][1:]

    # print(created_c)

    
    # here logic for temp_c/ : split on / and take the last one in source_asm.split
    # c_out_file_name = "temp_c/" + source_asm.split('/')[-1].split(".")[0] + "_out.c"
    
    # with open(c_out_file_name, "w") as file:
        # file.write(created_c)


    # 1.B) Assembly to .dot CFG directly
    source_asm, function_name = objToS("temp_c/",source_obj)

    c_out_file_name = "temp_c/" + source_asm.split('/')[-1].split(".")[0] + "_out.dot"
    cfg_file_name = asmToCFG(c_out_file_name, source_obj, function_name)

    

    # 2) C Code to CFG
    # not needed when using r2
    # cfg_file_name = cToCfg(c_out_file_name)

    # 3) CFG KNN search in NEO4j Database
    for filename in os.listdir("temp_c"):
        # Check if the file ends with .cfg
        if filename.endswith(".cfg"):
            cfg_file_name = filename


    matches = KNN_search.make_search("temp_c/"+cfg_file_name)

    # print(matches[0])

    # 4) Original Assmebly and CFG are fed to specified model for final C code decompile

    input_text = "Decompile the following assembly code to C code:"

    with open(source_asm, "r") as file:
        asm_function = file.read()
        input_asm_code = asm_function

    input = (
        "Assembly Code:\n" + input_asm_code + "\n\nDecompile the above Assembly Code into C code with the C code surrounded by ``` to indicate where the code is using the following reference C code as a similar structural reference\n\n" + "\n\n".join(matches)
    ) 
    # print("\n\n\n\n+++++++++++++++++++++++++++++++++++++++")

    results = querryModel(input)
    print(results)

    results = results.split("```")[-2][1:]
    # results = querryModel(input)

    print(results)
    return(results)

    # 5) Similarity Matching with SMT to determine effectiveness

if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    loadModel()
    main()