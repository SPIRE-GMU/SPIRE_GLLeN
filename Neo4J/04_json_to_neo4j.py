#!/usr/bin/env python3
import os
import json
from neo4j import GraphDatabase

# Neo4j connection details
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "rootboot"  # Update with your actual password

# Directory of JSON files produced by parsing the .dot files
JSON_DIRECTORY = "/home/spire2/SPIRE_GLLeN/Neo4J/json_files"

# Directories for raw .dot, .cfg, .asm, and .c files
DOT_DIRECTORY = "/home/spire2/SPIRE_GLLeN/Neo4J/dot_files_newDB"
CFG_DIRECTORY = "/home/spire2/SPIRE_GLLeN/Neo4J/cfg_files_newDB"
ASM_DIRECTORY = "/home/spire2/SPIRE_GLLeN/Neo4J/asm_files_newDB"
C_DIRECTORY   = "/home/spire2/SPIRE_GLLeN/Neo4J/c_files_train_real_compilable_newDB"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

###############################################################################
# Helper function: Read a file's contents or return "" if not found
###############################################################################
def read_file_if_exists(file_path):
    if os.path.isfile(file_path):
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    return ""

###############################################################################
# Main insertion function
###############################################################################
def insert_dot_json_into_neo4j(session, dot_json):
    """
    dot_json example structure:
    {
      "function_name": "A_is_valid",
      "nodes": [
        { "id": "fn_0_basic_block_2",  "attributes": {...} },
        ...
      ],
      "edges": [
        { "from": "fn_0_basic_block_2", "to": "fn_0_basic_block_3", "attributes": {...} },
        ...
      ]
    }
    We will:
      1) MERGE a :Function node for function_name
      2) Read raw .dot, .cfg, .asm, .c from the known directories using function_name
      3) Insert each node as :BasicBlock {id: <function_name>::<block_id>} 
         and link it to the :Function with (:Function)-[:HAS_BLOCK]->(:BasicBlock).
      4) Insert :NEXT edges, also referencing the new unique IDs.
    """

    function_name = dot_json.get("function_name", "UnknownFunc")

    # 1) Try to read raw file contents from .dot, .cfg, .asm, .c
    dot_path = os.path.join(DOT_DIRECTORY,  function_name + ".dot")
    cfg_path = os.path.join(CFG_DIRECTORY,  function_name + ".cfg")
    asm_path = os.path.join(ASM_DIRECTORY,  function_name + ".s")  # if files are .s
    c_path   = os.path.join(C_DIRECTORY,    function_name + ".c")

    dot_data = read_file_if_exists(dot_path)
    cfg_data = read_file_if_exists(cfg_path)
    asm_data = read_file_if_exists(asm_path)
    c_data   = read_file_if_exists(c_path)

    # 2) MERGE the :Function node, storing raw text
    session.run(
        """
        MERGE (f:Function {name: $function_name})
        SET f.dot_data = $dot_data,
            f.cfg_data = $cfg_data,
            f.asm_data = $asm_data,
            f.c_data   = $c_data
        """,
        function_name=function_name,
        dot_data=dot_data,
        cfg_data=cfg_data,
        asm_data=asm_data,
        c_data=c_data
    )

    # 3) Insert BasicBlock nodes
    #    We create a unique ID that combines function_name and the local block id
    #    to avoid collisions across different functions
    for node_info in dot_json.get("nodes", []):
        local_id = node_info["id"]  # e.g. "fn_0_basic_block_2"
        attr_dict = node_info.get("attributes", {})

        block_unique_id = f"{function_name}::{local_id}"  # ensure uniqueness
        block_label = attr_dict.get("label", local_id)

        session.run(
            """
            MATCH (f:Function {name: $function_name})
            MERGE (b:BasicBlock {id: $block_unique_id})
            SET b += $all_attrs
            SET b.label = $block_label
            MERGE (f)-[:HAS_BLOCK]->(b)
            """,
            function_name=function_name,
            block_unique_id=block_unique_id,
            all_attrs=attr_dict,
            block_label=block_label
        )

    # 4) Insert Edges (NEXT relationships)
    #    Must also reference the combined function_name + block ID
    for edge_info in dot_json.get("edges", []):
        src_local_id = edge_info["from"]
        dst_local_id = edge_info["to"]
        edge_attrs   = edge_info.get("attributes", {})

        src_unique = f"{function_name}::{src_local_id}"
        dst_unique = f"{function_name}::{dst_local_id}"

        session.run(
            """
            MATCH (src:BasicBlock {id: $src_unique})
            MATCH (dst:BasicBlock {id: $dst_unique})
            MERGE (src)-[n:NEXT]->(dst)
            SET n += $edge_attrs
            """,
            src_unique=src_unique,
            dst_unique=dst_unique,
            edge_attrs=edge_attrs
        )

###############################################################################
# main: iterate over JSON files, insert them into Neo4j
###############################################################################
def main():
    json_files = [f for f in os.listdir(JSON_DIRECTORY) if f.endswith(".json")]
    if not json_files:
        print(f"[WARNING] No JSON files found in {JSON_DIRECTORY}")
        return

    with driver.session() as session:
        for jf in json_files:
            json_path = os.path.join(JSON_DIRECTORY, jf)
            with open(json_path, "r", encoding="utf-8") as f:
                dot_json = json.load(f)

            func_name = dot_json.get("function_name")
            print(f"Inserting {jf} -> function_name={func_name}")
            insert_dot_json_into_neo4j(session, dot_json)

    print("All DOT-based JSON data has been successfully inserted into Neo4j.")

if __name__ == "__main__":
    main()
