#!/usr/bin/env python3
import sys
import os
import re
import uuid
import json
import torch
import numpy as np

from transformers import RobertaTokenizer, RobertaModel
from neo4j import GraphDatabase
from sklearn.metrics.pairwise import cosine_similarity

###############################################################################
# Configuration
###############################################################################
# Neo4j connection details
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "rootboot"  

# If you want to optionally store the generated JSON
OUTPUT_JSON_DIR = "/home/spire2/SPIRE_GLLeN/Neo4J/temp_files"
os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)

# Path to CodeBERT model
CODEBERT_MODEL_NAME = "microsoft/codebert-base"

###############################################################################
# 1) Set up CodeBERT
###############################################################################
print("[INFO] Loading CodeBERT...")
tokenizer = RobertaTokenizer.from_pretrained(CODEBERT_MODEL_NAME)
model = RobertaModel.from_pretrained(CODEBERT_MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print("[INFO] CodeBERT loaded.")

###############################################################################
# 2) Connect to Neo4j
###############################################################################
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

###############################################################################
# SIMPLE EMBEDDING FUNCTION (Structural)
###############################################################################
def compute_structural_embedding(nodes, edges, loops_count):
    """
    If you want a minimal structural feature vector, here it is.
    Currently unused for the CodeBERT approach, but kept as a reference.
    """
    return [len(nodes), len(edges), loops_count]

###############################################################################
# HELPER: READ .c FILE FOR A GIVEN .CFG
###############################################################################
def read_c_code_for_cfg(cfg_filename, c_files_dir="/home/spire2/SPIRE_GLLeN/Neo4J/c_files"):
    """
    If your .cfg is 'foo.c.cfg', we try to read 'foo.c' from c_files_dir
    Return the entire .c file content, or empty string if not found.
    """
    base_name = os.path.splitext(cfg_filename)[0]  # e.g. 'foo.c' from 'foo.c.cfg'
    if base_name.endswith(".c"):
        short_base = os.path.splitext(base_name)[0]  # e.g. 'foo'
    else:
        short_base = base_name

    c_file_candidate = short_base + ".c"
    c_file_path = os.path.join(c_files_dir, c_file_candidate)

    if os.path.isfile(c_file_path):
        try:
            with open(c_file_path, "r", encoding="utf-8", errors="replace") as cf:
                return cf.read()
        except:
            return ""
    else:
        return ""

###############################################################################
# PARSE A SINGLE .CFG FILE => Return a dictionary with function_name, nodes, etc.
###############################################################################
def parse_cfg_file(cfg_file_path):
    with open(cfg_file_path, 'r', encoding="utf-8", errors="replace") as f:
        cfg_lines = f.readlines()
    cfg_entire_text = "".join(cfg_lines)

    # Attempt to read function_name from lines with ';; Function <name>'
    function_name = None
    for line in cfg_lines:
        line_stripped = line.strip()
        if line_stripped.startswith(";; Function"):
            match = re.match(r";; Function (\w+)", line_stripped)
            if match:
                function_name = match.group(1)
                break
    if not function_name:
        function_name = "UnknownFunc"

    # Create a sanitized prefix
    function_name_sanitized = re.sub(r"[^a-zA-Z0-9]+", "_", function_name)
    random_suffix = str(uuid.uuid4())[:8]
    prefix = f"{function_name_sanitized}-{random_suffix}"

    cfg_basename = os.path.basename(cfg_file_path)

    block_id_map = {}
    nodes = []
    edges = []
    loops_count = 0

    function_code_accumulator = []
    current_block_id = None
    current_block_code = []

    end_node_id = f"{prefix}-end"
    end_node_used = False

    for line in cfg_lines:
        line_stripped = line.strip()
        # check loops
        loops_match = re.match(r";;\s*(\d+)\s+loops found", line_stripped)
        if loops_match:
            loops_count = int(loops_match.group(1))

        # detect <bb X>
        if line_stripped.startswith("<bb"):
            # finalize previous block
            if current_block_id is not None:
                unique_block_id = block_id_map[current_block_id]
                block_label = f"BB {current_block_id}"
                block_code_str = "\n".join(current_block_code)
                nodes.append({
                    "id": unique_block_id,
                    "label": block_label,
                    "block_num": current_block_id,
                    "code": block_code_str
                })
            match_bb = re.match(r"<bb (\d+)>", line_stripped)
            if match_bb:
                block_num = int(match_bb.group(1))
                if block_num == 1:
                    current_block_id = None
                    current_block_code = []
                else:
                    current_block_id = block_num
                    block_uuid = f"{prefix}-bb-{block_num}"
                    block_id_map[block_num] = block_uuid
                    current_block_code = [line_stripped]
            else:
                current_block_id = None
                current_block_code = []

        elif "succs" in line_stripped:
            m_succs = re.match(r";;\s*(\d+)\s+succs\s+\{\s*([\d\s]+)\s*\}", line_stripped)
            if m_succs:
                src_block_num = int(m_succs.group(1))
                if src_block_num in block_id_map:
                    src_block_uuid = block_id_map[src_block_num]
                else:
                    if src_block_num == 1:
                        src_block_uuid = end_node_id
                        end_node_used = True
                    else:
                        src_block_uuid = f"{prefix}-bb-{src_block_num}"
                        block_id_map[src_block_num] = src_block_uuid

                succ_str = m_succs.group(2).strip()
                succ_list = [s for s in succ_str.split() if s.isdigit()]
                for sblk in succ_list:
                    tgt_num = int(sblk)
                    if tgt_num == 1:
                        edges.append({"from": src_block_uuid, "to": end_node_id})
                        end_node_used = True
                    else:
                        if tgt_num not in block_id_map:
                            block_id_map[tgt_num] = f"{prefix}-bb-{tgt_num}"
                        tgt_uuid = block_id_map[tgt_num]
                        edges.append({"from": src_block_uuid, "to": tgt_uuid})

        else:
            # accumulate lines
            if current_block_id is not None:
                current_block_code.append(line_stripped)
            else:
                function_code_accumulator.append(line_stripped)

    # finalize last block if any
    if current_block_id is not None:
        unique_block_id = block_id_map[current_block_id]
        block_label = f"BB {current_block_id}"
        block_code_str = "\n".join(current_block_code)
        nodes.append({
            "id": unique_block_id,
            "label": block_label,
            "block_num": current_block_id,
            "code": block_code_str
        })

    # read the .c file content
    c_file_content = read_c_code_for_cfg(cfg_basename)
    function_label = f"Function: {function_name}"
    func_node_id = f"{prefix}-func"

    # add the function node
    nodes.append({
        "id": func_node_id,
        "label": function_label,
        "function_name": function_name,
        "cfg_filename": cfg_basename,
        "block_num": None,
        "code": "\n".join(function_code_accumulator),
        "cfg_data": cfg_entire_text,
        "c_data": c_file_content
    })

    # if block2 exists, link function node to it
    if 2 in block_id_map:
        edges.append({"from": func_node_id, "to": block_id_map[2]})

    # if end node used, add it
    if end_node_used:
        nodes.append({
            "id": end_node_id,
            "label": "End",
            "block_num": 1,
            "code": ""
        })

    cfg_id = f"{function_name}-{prefix}"

    parsed_data = {
        "cfg_id": cfg_id,
        "function_name": function_name,
        "cfg_filename": cfg_basename,
        "nodes": nodes,
        "edges": edges,
        "loops_count": loops_count
    }
    return parsed_data

###############################################################################
# CodeBERT EMBEDDING
###############################################################################
def get_codebert_embedding(code_str, max_len=512):
    """
    Generate a 768-dim embedding from the CodeBERT model for the given code string.
    """
    from transformers import RobertaTokenizer, RobertaModel
    inputs = tokenizer(code_str, max_length=max_len, truncation=True, return_tensors='pt')
    for k, v in inputs.items():
        inputs[k] = v.to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        cls_vec = outputs.last_hidden_state[0, 0, :]
    return cls_vec.cpu().numpy()

###############################################################################
# LOAD DB EMBEDDINGS & KNN SEARCH
###############################################################################
def load_db_embeddings():
    """
    Load all (function_name, codebertEmbedding) from Neo4j
    Return list of (function_name, np.array of shape [768])
    """
    with driver.session() as session:
        result = session.run("""
            MATCH (f:Function)
            WHERE f.codebertEmbedding IS NOT NULL
            RETURN f.function_name AS fname, f.codebertEmbedding AS emb
        """)
        data = []
        for rec in result:
            fname = rec["fname"]
            emb = rec["emb"]  # a list of floats
            data.append((fname, np.array(emb)))  # convert to np array
    return data

def compute_knn(new_vec, db_data, top_k=3):
    """
    new_vec: shape [768]
    db_data: list of (fname, np.array[768])
    Return top_k matches by descending cosine similarity
    """
    from sklearn.metrics.pairwise import cosine_similarity
    new_vec = new_vec.reshape(1, -1)
    names = []
    all_vecs = []
    for (fname, e) in db_data:
        names.append(fname)
        all_vecs.append(e)
    all_vecs = np.vstack(all_vecs)  # shape [N, 768]

    sims = cosine_similarity(new_vec, all_vecs)[0]  # shape [N]
    idx_sorted = np.argsort(-sims)
    top_idx = idx_sorted[:top_k]
    results = []
    for i in top_idx:
        results.append((names[i], sims[i]))
    return results

###############################################################################
# MAIN
###############################################################################
def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {os.path.basename(__file__)} <new.cfg>")
        sys.exit(1)

    cfg_file_path = sys.argv[1]
    if not os.path.isfile(cfg_file_path):
        print(f"[ERROR] The file '{cfg_file_path}' does not exist.")
        sys.exit(1)

    # 1) Parse the .cfg => produce a dictionary
    parsed = parse_cfg_file(cfg_file_path)

    # 2) Optionally write the JSON for reference
    base_name = os.path.splitext(os.path.basename(cfg_file_path))[0]
    output_json_path = os.path.join(OUTPUT_JSON_DIR, base_name + ".json")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "cfg_id": parsed["cfg_id"],
            "function_name": parsed["function_name"],
            "nodes": parsed["nodes"],
            "edges": parsed["edges"],
            "loops_count": parsed["loops_count"]
        }, f, indent=2)
    print(f"[INFO] Created JSON: {output_json_path}")

    # 3) Find the function node => block_num=null => code
    nodes = parsed["nodes"]
    func_node = next((n for n in nodes if n.get("block_num") is None), None)
    if not func_node:
        print("[ERROR] No function node (block_num=null) found in parsed CFG. Exiting.")
        sys.exit(1)

    code_str = func_node.get("code", "")
    if not code_str:
        print("[ERROR] The function node code is empty. Exiting.")
        sys.exit(1)

    # 4) Generate CodeBERT embedding
    print("[INFO] Generating CodeBERT embedding for new CFG code...")
    new_emb = get_codebert_embedding(code_str)  # shape [768,]

    # 5) Load existing embeddings from DB
    print("[INFO] Loading existing codebertEmbedding from Neo4j...")
    db_data = load_db_embeddings()
    if not db_data:
        print("[WARN] No existing embeddings found in DB. Exiting.")
        sys.exit(0)

    # 6) KNN search
    print("[INFO] Performing KNN search via cosine similarity...")
    top_matches = compute_knn(new_emb, db_data, top_k=10)

    # 7) Print results
    print("\n[RESULT] Top 3 matches by similarity:")
    for (fname, score) in top_matches:
        print(f" - {fname}: similarity={score:.4f}")

    driver.close()
    print("[INFO] Done.")

if __name__ == "__main__":
    main()
