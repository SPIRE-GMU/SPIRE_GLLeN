#!/usr/bin/env python3
import re
import uuid
import os
import json

###############################################################################
# Configuration
###############################################################################
# Directory containing your .cfg files
CFG_DIRECTORY = "/home/spire2/SPIRE_GLLeN/Neo4J/cfg_files"

# Directory containing your corresponding .c files
C_FILES_DIRECTORY = "/home/spire2/SPIRE_GLLeN/Neo4J/c_files"

# Output directory for JSON
OUTPUT_JSON_DIR = "/home/spire2/SPIRE_GLLeN/Neo4J/json_files"
os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)


###############################################################################
# Simple Embedding Function
###############################################################################
def compute_embedding(nodes, edges, loops_count):
    """
    Given a list of node dicts, edge dicts, and the number of loops,
    produce a numeric embedding for KNN.

    We return [num_nodes, num_edges, loops_count].
    """
    return [len(nodes), len(edges), loops_count]


###############################################################################
# Helper: Read matching .c file
###############################################################################
def read_c_code_for_cfg(cfg_filename):
    """
    Given the base name of the .cfg file, try to locate a corresponding .c file
    in C_FILES_DIRECTORY.

    Example:
      if cfg_filename = "foo.c.cfg", then base is "foo.c"
      we try to read: /home/spire2/SPIRE_GLLeN/Neo4J/c_files/foo.c

    If that doesn't exist, we return an empty string.
    """
    base_name = os.path.splitext(cfg_filename)[0]  # e.g. "foo.c" from "foo.c.cfg"

    if base_name.endswith(".c"):
        # e.g. "foo.c" -> remove .c -> "foo"
        short_base = os.path.splitext(base_name)[0]
    else:
        short_base = base_name

    # We'll guess the .c file is short_base + ".c"
    c_file_candidate = short_base + ".c"
    c_file_path = os.path.join(C_FILES_DIRECTORY, c_file_candidate)

    if os.path.isfile(c_file_path):
        try:
            with open(c_file_path, "r", encoding="utf-8", errors="replace") as cf:
                return cf.read()
        except:
            return ""
    else:
        return ""


###############################################################################
# Parse a Single .cfg File
###############################################################################
def parse_cfg_file(file_path):
    """
    Reads lines from the .cfg file and extracts:
      - function name
      - basic blocks (bb)
      - edges (succs lines)
      - loops count (via ";; X loops found")

    Every graph should:
      1) Start with a function node that points to BB2 (if BB2 exists)
      2) If an edge points to BB1, treat that as an End node instead of a real block

    Additionally:
      - We'll store the entire CFG text in the function node as 'cfg_data'.
      - We'll store the entire corresponding .c code in the function node as 'c_data'.
      - We'll store extra properties ("function_name", "cfg_filename") for convenience.
      - We'll prefix all node IDs with a sanitized version of the function name.

    Returns a dict with:
      {
        "cfg_id": <string>,
        "function_name": <string or None>,
        "cfg_filename": <string>,
        "nodes": [
          {
            "id": <unique string id>,
            "label": <"BB 2" or "Function: foo">,
            "block_num": <int or None>,
            "code": <string code>,
            ...
          },
          ...
        ],
        "edges": [ {"from": ..., "to": ...}, ...],
        "loops_count": <int>
      }
    """
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        cfg_lines = f.readlines()
    cfg_entire_text = "".join(cfg_lines)

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

    function_name_sanitized = re.sub(r"[^a-zA-Z0-9]+", "_", function_name)
    random_suffix = str(uuid.uuid4())[:8]
    prefix = f"{function_name_sanitized}-{random_suffix}"

    cfg_basename = os.path.basename(file_path)
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

        # Check for loop count
        loops_match = re.match(r";;\s*(\d+)\s+loops found", line_stripped)
        if loops_match:
            loops_count = int(loops_match.group(1))

        # Basic block start
        if line_stripped.startswith("<bb"):
            if current_block_id is not None:
                unique_block_id = block_id_map[current_block_id]
                block_label = f"BB {current_block_id}"
                block_code_str = "\n".join(current_block_code)
                nodes.append(
                    {
                        "id": unique_block_id,
                        "label": block_label,
                        "block_num": current_block_id,
                        "code": block_code_str,
                    }
                )

            match_block = re.match(r"<bb (\d+)>", line_stripped)
            if match_block:
                block_num = int(match_block.group(1))
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
            match_succs = re.match(
                r";;\s*(\d+)\s+succs\s+\{\s*([\d\s]+)\s*\}", line_stripped
            )
            if match_succs:
                src_block_num = int(match_succs.group(1))
                if src_block_num in block_id_map:
                    src_block_uuid = block_id_map[src_block_num]
                else:
                    if src_block_num == 1:
                        src_block_uuid = end_node_id
                        end_node_used = True
                    else:
                        src_block_uuid = f"{prefix}-bb-{src_block_num}"
                        block_id_map[src_block_num] = src_block_uuid

                succ_str = match_succs.group(2).strip()
                succ_list = [s for s in succ_str.split() if s.isdigit()]

                for succ_block_str in succ_list:
                    succ_block_num = int(succ_block_str)
                    if succ_block_num == 1:
                        edges.append({"from": src_block_uuid, "to": end_node_id})
                        end_node_used = True
                    else:
                        if succ_block_num not in block_id_map:
                            block_id_map[succ_block_num] = (
                                f"{prefix}-bb-{succ_block_num}"
                            )
                        succ_block_uuid = block_id_map[succ_block_num]
                        edges.append({"from": src_block_uuid, "to": succ_block_uuid})

        else:
            if current_block_id is not None:
                current_block_code.append(line_stripped)
            else:
                function_code_accumulator.append(line_stripped)

    if current_block_id is not None:
        unique_block_id = block_id_map[current_block_id]
        block_label = f"BB {current_block_id}"
        block_code_str = "\n".join(current_block_code)
        nodes.append(
            {
                "id": unique_block_id,
                "label": block_label,
                "block_num": current_block_id,
                "code": block_code_str,
            }
        )

    c_file_content = read_c_code_for_cfg(cfg_basename)
    function_label = f"Function: {function_name}"
    func_node_id = f"{prefix}-func"
    nodes.append(
        {
            "id": func_node_id,
            "label": function_label,
            "function_name": function_name,
            "cfg_filename": cfg_basename,
            "block_num": None,
            "code": "\n".join(function_code_accumulator),
            "cfg_data": cfg_entire_text,
            "c_data": c_file_content,
        }
    )

    if 2 in block_id_map:
        edges.append({"from": func_node_id, "to": block_id_map[2]})

    if end_node_used:
        nodes.append({"id": end_node_id, "label": "End", "block_num": 1, "code": ""})

    cfg_id = f"{function_name}-{prefix}"

    return {
        "cfg_id": cfg_id,
        "function_name": function_name,
        "cfg_filename": cfg_basename,
        "nodes": nodes,
        "edges": edges,
        "loops_count": loops_count,
    }


###############################################################################
# Convert CFG -> JSON
###############################################################################
def process_single_cfg_file(cfg_file_path):
    parsed_data = parse_cfg_file(cfg_file_path)

    nodes = parsed_data["nodes"]
    edges = parsed_data["edges"]
    loops_count = parsed_data["loops_count"]

    # Build the embedding with (num_nodes, num_edges, loops_count)
    embedding = compute_embedding(nodes, edges, loops_count)

    final_dict = {
        "cfg_id": parsed_data["cfg_id"],
        "function_name": parsed_data["function_name"],
        "cfg_filename": parsed_data["cfg_filename"],
        "nodes": nodes,
        "edges": edges,
        "embedding": embedding,
    }

    # Check if the function has more than 50 nodes
    node_count = len(nodes)
    if node_count > 50:
        print(
            f"[ALERT] Large function detected in {cfg_file_path} with {node_count} nodes."
        )

    base_name = os.path.splitext(os.path.basename(cfg_file_path))[0]
    output_json_path = os.path.join(OUTPUT_JSON_DIR, f"{base_name}.json")

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(final_dict, f, indent=2)

    # print(f"[INFO] Created JSON: {output_json_path}")


def main():
    cfg_files = [f for f in os.listdir(CFG_DIRECTORY) if f.endswith(".cfg")]
    if not cfg_files:
        print(f"[WARNING] No .cfg files found in {CFG_DIRECTORY}")
        return

    for cfg_file in cfg_files:
        cfg_path = os.path.join(CFG_DIRECTORY, cfg_file)
        process_single_cfg_file(cfg_path)


if __name__ == "__main__":
    main()
