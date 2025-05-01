#!/usr/bin/env python3
import os
import json
import re
from neo4j import GraphDatabase

# Neo4j connection details
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "rootboot"  # update if needed

# Directories for JSON and raw files
JSON_DIRECTORY = "/home/spire2/SPIRE_GLLeN/Neo4J/justin/json_files"
DOT_DIRECTORY = "/home/spire2/SPIRE_GLLeN/Neo4J/justin/dot_files_newDB"
CFG_DIRECTORY = "/home/spire2/SPIRE_GLLeN/Neo4J/justin/cfg_files_newDB"
ASM_DIRECTORY = "/home/spire2/SPIRE_GLLeN/Neo4J/justin/asm_files_newDB"
C_DIRECTORY = (
    "/home/spire2/SPIRE_GLLeN/Neo4J/justin/c_files_train_real_compilable_newDB"
)

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def read_file_if_exists(file_path):
    if os.path.isfile(file_path):
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    return ""


def extract_block_number(block_id):
    """Attempt to parse 'basic_block_(X)' from block_id, or return original."""
    match = re.search(r"basic_block_(\d+)", block_id)
    return match.group(1) if match else block_id


def insert_dot_json_into_neo4j(session, dot_json, json_filename):
    """
    Ingests one dot_json structure into Neo4j.

    We use the json_filename (minus extension) both as:
      - The unique_id for the :Function node
      - The base for reading .dot/.cfg/.s/.c
    Then for each BasicBlock, we store:
      b.function_unique_id = <base_name>
      b.function_name = <function_name from JSON>
    so it's clear which function they belong to.
    """
    base_name = os.path.splitext(json_filename)[0]  # e.g. 'main_1763'
    # The function_name from the JSON is mostly for UI reference
    function_name = dot_json.get("function_name", "UnknownFunc")

    # File paths based on base_name
    dot_path = os.path.join(DOT_DIRECTORY, base_name + ".dot")
    cfg_path = os.path.join(CFG_DIRECTORY, base_name + ".cfg")
    asm_path = os.path.join(ASM_DIRECTORY, base_name + ".s")
    c_path = os.path.join(C_DIRECTORY, base_name + ".c")

    dot_data = read_file_if_exists(dot_path)
    cfg_data = read_file_if_exists(cfg_path)
    asm_data = read_file_if_exists(asm_path)
    c_data = read_file_if_exists(c_path)

    # Create or MERGE the Function node by property unique_id=base_name
    session.run(
        """
        MERGE (f:Function {unique_id: $unique_id})
        SET f.name      = $function_name,
            f.dot_data  = $dot_data,
            f.cfg_data  = $cfg_data,
            f.asm_data  = $asm_data,
            f.c_data    = $c_data
        """,
        unique_id=base_name,
        function_name=function_name,
        dot_data=dot_data,
        cfg_data=cfg_data,
        asm_data=asm_data,
        c_data=c_data,
    )

    # Insert BasicBlocks
    entry_node_id = None
    for node_info in dot_json.get("nodes", []):
        local_id = node_info["id"]
        attr_dict = node_info.get("attributes", {})

        # Build block's unique ID (function's base_name + local_id)
        block_unique_id = f"{base_name}::{local_id}"

        label_str = attr_dict.get("label", "")
        if "ENTRY" in label_str:
            node_name = "ENTRY"
            entry_node_id = block_unique_id
        elif "EXIT" in label_str:
            node_name = "EXIT"
        else:
            node_name = extract_block_number(local_id)

        # Here we store function_unique_id and function_name on the block
        session.run(
            """
            MERGE (b:BasicBlock {id: $block_id})
            SET b += $all_attrs,
                b.name = $node_name,
                b.function_unique_id = $unique_id,
                b.function_name = $function_name
            """,
            block_id=block_unique_id,
            all_attrs=attr_dict,
            node_name=node_name,
            unique_id=base_name,
            function_name=function_name,
        )

    # Link Function → ENTRY block
    if entry_node_id:
        session.run(
            """
            MATCH (f:Function {unique_id: $unique_id}), (b:BasicBlock {id: $entry_id})
            MERGE (f)-[:STARTS_AT]->(b)
            """,
            unique_id=base_name,
            entry_id=entry_node_id,
        )

    # Insert NEXT edges
    for edge_info in dot_json.get("edges", []):
        src_local = edge_info["from"]
        dst_local = edge_info["to"]
        edge_attrs = edge_info.get("attributes", {})

        # skip invisible edge from block_0 -> block_1
        if (
            src_local.endswith("basic_block_0")
            and dst_local.endswith("basic_block_1")
            and edge_attrs.get("style", "") == "invis"
        ):
            continue

        src_id = f"{base_name}::{src_local}"
        dst_id = f"{base_name}::{dst_local}"

        session.run(
            """
            MATCH (src:BasicBlock {id: $src_id})
            MATCH (dst:BasicBlock {id: $dst_id})
            MERGE (src)-[e:NEXT]->(dst)
            SET e += $edge_attrs
            """,
            src_id=src_id,
            dst_id=dst_id,
            edge_attrs=edge_attrs,
        )


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

            func_name = dot_json.get("function_name", "[NO FUNC NAME]")
            print(f"[→] Inserting {jf} (function_name={func_name})")
            insert_dot_json_into_neo4j(session, dot_json, jf)

    print(
        "✅ All DOT-based JSON data has been successfully inserted into Neo4j with function_unique_id on each BasicBlock."
    )


if __name__ == "__main__":
    main()
