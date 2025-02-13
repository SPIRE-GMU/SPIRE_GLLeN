#!/usr/bin/env python3
import os
import json
from neo4j import GraphDatabase

# Neo4j connection details
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "rootboot"  # update with your actual password

# Directory of JSON files
JSON_DIRECTORY = "/home/spire2/SPIRE_GLLeN/Neo4J/json_files"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def insert_one_cfg_into_neo4j(session, cfg_json):
    cfg_id = cfg_json["cfg_id"]
    function_name = cfg_json.get("function_name", None)
    cfg_filename = cfg_json.get("cfg_filename", None)
    embedding = cfg_json.get("embedding", [])
    nodes = cfg_json["nodes"]
    edges = cfg_json["edges"]

    # Find the function node (block_num == None)
    function_node = next((n for n in nodes if n.get("block_num") is None), None)
    if not function_node:
        print(f"[WARNING] No function node found for cfg_id={cfg_id}; skipping.")
        return

    func_id = function_node["id"]
    func_label = function_node["label"]         # e.g. "Function: main"
    func_code = function_node.get("code", "")
    cfg_data = function_node.get("cfg_data", "")
    c_data = function_node.get("c_data", "")

    # Create or MERGE the :Function node, setting "name" = func_label
    session.run(
        """
        MERGE (f:Function {id: $func_id})
        SET f.cfg_id        = $cfg_id,
            f.function_name = $function_name,
            f.cfg_filename  = $cfg_filename,
            f.label         = $func_label,
            f.name          = $func_label,        // <-- Set 'name' for UI
            f.code          = $func_code,
            f.cfg_data      = $cfg_data,
            f.c_data        = $c_data,
            f.embedding     = $embedding
        """,
        func_id=func_id,
        cfg_id=cfg_id,
        function_name=function_name,
        cfg_filename=cfg_filename,
        func_label=func_label,
        func_code=func_code,
        cfg_data=cfg_data,
        c_data=c_data,
        embedding=embedding
    )

    # Create or MERGE the other nodes (BasicBlock or End)
    basic_blocks = [n for n in nodes if n.get("block_num") is not None]
    for block in basic_blocks:
        block_id = block["id"]
        block_num = block["block_num"]
        label = block["label"]       # e.g. "BB 2" or "End"
        code = block.get("code", "")

        if block_num == 1 and label == "End":
            # For the End node, label it :End
            session.run(
                """
                MERGE (e:End {id: $block_id})
                SET e.block_num = $block_num,
                    e.label     = $label,
                    e.name      = $label,   // Also set 'name'
                    e.code      = $code
                """,
                block_id=block_id,
                block_num=block_num,
                label=label,
                code=code
            )
        else:
            # For normal basic blocks
            session.run(
                """
                MERGE (b:BasicBlock {id: $block_id})
                SET b.block_num = $block_num,
                    b.label     = $label,
                    b.name      = $label,   // Also set 'name'
                    b.code      = $code
                """,
                block_id=block_id,
                block_num=block_num,
                label=label,
                code=code
            )

    # Create or MERGE edges
    for edge in edges:
        src_id = edge["from"]
        dst_id = edge["to"]
        session.run(
            """
            MATCH (src {id: $src_id}), (dst {id: $dst_id})
            MERGE (src)-[:NEXT]->(dst)
            """,
            src_id=src_id,
            dst_id=dst_id
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
                cfg_json = json.load(f)
            insert_one_cfg_into_neo4j(session, cfg_json)

    print("All CFG data has been successfully inserted into Neo4j.")

if __name__ == "__main__":
    main()
