#!/usr/bin/env python3
import os
import json
import torch
from neo4j import GraphDatabase
from transformers import RobertaTokenizer, RobertaModel

# Neo4j connection details
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "rootboot"  # adjust with your actual password

# Directory with JSON files
JSON_DIR = "/home/spire2/SPIRE_GLLeN/Neo4J/json_files"

# Load CodeBERT model and tokenizer
model_name = "microsoft/codebert-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaModel.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Connect to Neo4j
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def get_codebert_embedding(code_str, max_len=512):
    """
    Given a code snippet (string), return a 768-d CodeBERT embedding as a Python list.
    """
    inputs = tokenizer(
        code_str,
        max_length=max_len,
        truncation=True,
        return_tensors='pt'
    )
    for k, v in inputs.items():
        inputs[k] = v.to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        # For a RoBERTa-based model, the first token is <s>:
        # We'll take outputs.last_hidden_state[0,0,:] as the embedding
        cls_vec = outputs.last_hidden_state[0, 0, :]  # shape [768]
    
    return cls_vec.cpu().numpy().tolist()

def store_embedding_in_neo4j(function_name, embedding_list):
    """
    Store the embedding on the :Function node that has function_name = $function_name.
    If your DB uses a different property (e.g. function_id), update the MATCH clause.
    """
    with driver.session() as session:
        result = session.run(
            """
            MATCH (f:Function {function_name: $fname})
            SET f.codebertEmbedding = $emb
            RETURN f.function_name AS name
            """,
            fname=function_name,
            emb=embedding_list
        )
        record = result.single()
        if record:
            print(f"[INFO] Stored embedding for function_name='{record['name']}' in Neo4j.")
        else:
            print(f"[WARN] No :Function node found in Neo4j with function_name='{function_name}'.")

def main():
    json_files = [f for f in os.listdir(JSON_DIR) if f.endswith(".json")]
    if not json_files:
        print(f"[WARN] No JSON files found in {JSON_DIR}.")
        return

    for jf in json_files:
        full_path = os.path.join(JSON_DIR, jf)
        with open(full_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Example JSON structure:
        # {
        #   "cfg_id": "...",
        #   "function_name": "...",
        #   "nodes": [
        #       { "block_num": null, "code": "full function code..." },
        #       ...
        #   ],
        #   ...
        # }
        #
        # We'll read function_name from top-level, then find the node with block_num=null for code.

        function_name = data.get("function_name", None)
        if not function_name:
            print(f"[WARN] {jf} has no 'function_name'. Skipping.")
            continue
        
        # Find the function node in "nodes" where block_num is null
        nodes = data.get("nodes", [])
        func_node = next((n for n in nodes if n.get("block_num") is None), None)
        if not func_node:
            print(f"[WARN] {jf} no node with block_num=null found. Skipping.")
            continue
        
        code_str = func_node.get("code", "")
        if not code_str:
            print(f"[WARN] {jf} function node is missing 'code'. Skipping.")
            continue
        
        # Compute CodeBERT embedding
        embedding = get_codebert_embedding(code_str)

        # Store in Neo4j
        store_embedding_in_neo4j(function_name, embedding)

    driver.close()
    print("[INFO] Done processing all JSON files.")

if __name__ == "__main__":
    main()
