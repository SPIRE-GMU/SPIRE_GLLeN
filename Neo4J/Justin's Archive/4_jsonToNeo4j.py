from neo4j import GraphDatabase
import json
import os

# Neo4j connection details
NEO4J_URI = "bolt://localhost:7687"  # Update if using a different host or port
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "rootboot"  # Update with the actual password

# Load the CFG JSON data
json_file_path = "/home/spire2/SPIRE_GLLeN/Neo4J/cfg_data.json"
with open(json_file_path, "r") as json_file:
    cfg_data = json.load(json_file)

# Connect to the Neo4j database
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def insert_data_into_neo4j(cfg_data):
    with driver.session() as session:
        for file_path, function_data in cfg_data.items():
            # Insert function node
            function_node = function_data["function_node"]

            # Load the `.s` and `.c` files from their respective directories
            function_name = function_node["function_name"]
            s_file_path = os.path.join(
                "/home/spire2/SPIRE_GLLeN/Neo4J/asm_files", f"{function_name}.s"
            )
            c_file_path = os.path.join(
                "/home/spire2/SPIRE_GLLeN/Neo4J/c_files", f"{function_name}.c"
            )

            s_file = ""
            c_file = ""

            # Read the content of the .s file
            if os.path.exists(s_file_path):
                with open(s_file_path, "r") as s_file_content:
                    s_file = s_file_content.read()

            # Read the content of the .c file
            if os.path.exists(c_file_path):
                with open(c_file_path, "r") as c_file_content:
                    c_file = c_file_content.read()

            session.run(
                """
                MERGE (f:Function {function_id: $function_id})
                SET f.function_name = $function_name, 
                    f.full_code = $full_code, 
                    f.name = $function_name,
                    f.s_file = $s_file,
                    f.c_file = $c_file
                """,
                function_id=function_node["function_id"],
                function_name=function_node["function_name"],
                full_code=function_node["full_code"],
                s_file=s_file,
                c_file=c_file,
            )

            # Insert basic blocks
            for block_id, block_data in function_data["basic_blocks"].items():
                session.run(
                    """
                    MERGE (b:BasicBlock {block_id: $block_id})
                    SET b.block_code = $block_code, 
                        b.contains_decision = $contains_decision,
                        b.is_loop_header = $is_loop_header, 
                        b.name = 'BB-' + split($block_id, '-')[3]
                    """,
                    block_id=block_data["block_id"],
                    block_code=block_data["block_code"],
                    contains_decision=block_data["contains_decision"],
                    is_loop_header=block_data["is_loop_header"],
                )

            # Insert basic block 2 relationship after all basic blocks are created
            starting_block = next(
                (
                    block_id
                    for block_id, block_data in function_data["basic_blocks"].items()
                    if block_id.endswith("-bb-2")
                ),
                None,
            )
            if starting_block:
                session.run(
                    """
                    MATCH (f:Function {function_id: $function_id}), (b:BasicBlock {block_id: $block_id})
                    MERGE (f)-[:STARTS_AT]->(b)
                    """,
                    function_id=function_node["function_id"],
                    block_id=starting_block,
                )

            # Insert loops (skip Loop 0)
            for loop_data in function_data["loops"]:
                if loop_data["loop_id"].endswith("-0"):
                    continue  # Skip loop 0

                session.run(
                    """
                    MERGE (l:Loop {loop_id: $loop_id})
                    SET l.header = $header, 
                        l.latch = $latch, 
                        l.depth = $depth, 
                        l.name = 'Loop-' + split($loop_id, '-')[3]
                    """,
                    loop_id=loop_data["loop_id"],
                    header=loop_data["header"],
                    latch=loop_data["latch"],
                    depth=loop_data["depth"],
                )

                # Create relationships between loop and nodes within the loop
                loop_nodes = loop_data.get("nodes", [])
                for node_id in loop_nodes:
                    # Ensure that the basic block exists before attempting to create the relationship
                    session.run(
                        """
                        MATCH (l:Loop {loop_id: $loop_id}), (b:BasicBlock {block_id: $node_id})
                        MERGE (l)-[:CONTAINS]->(b)
                        """,
                        loop_id=loop_data["loop_id"],
                        node_id=node_id,
                    )

            # Insert decision nodes
            for decision_data in function_data["decision_nodes"]:
                session.run(
                    """
                    MERGE (d:Decision {decision_id: $decision_id})
                    SET d.condition = $condition, 
                        d.name = $condition
                    """,
                    decision_id=decision_data["decision_id"],
                    condition=decision_data["condition"],
                )

                # Create relationship from the basic block to the decision node
                session.run(
                    """
                    MATCH (b:BasicBlock {block_id: $block_id}), (d:Decision {decision_id: $decision_id})
                    MERGE (b)-[:POINTS_TO_DECISION]->(d)
                    """,
                    block_id=decision_data["block_id"],
                    decision_id=decision_data["decision_id"],
                )

                # Create decision relationships from the decision node to the true/false targets
                if decision_data["true_target"]:
                    session.run(
                        """
                        MATCH (d:Decision {decision_id: $decision_id}), (b:BasicBlock {block_id: $true_target})
                        MERGE (d)-[:TRUE_FLOW]->(b)
                        """,
                        decision_id=decision_data["decision_id"],
                        true_target=decision_data["true_target"],
                    )
                if decision_data["false_target"]:
                    session.run(
                        """
                        MATCH (d:Decision {decision_id: $decision_id}), (b:BasicBlock {block_id: $false_target})
                        MERGE (d)-[:FALSE_FLOW]->(b)
                        """,
                        decision_id=decision_data["decision_id"],
                        false_target=decision_data["false_target"],
                    )

            # Insert edges (control flow relationships between basic blocks)
            for edge_data in function_data["edges"]:
                session.run(
                    """
                    MATCH (b1:BasicBlock {block_id: $from_node}), (b2:BasicBlock {block_id: $to_node})
                    MERGE (b1)-[:NEXT]->(b2)
                    """,
                    from_node=edge_data["from"],
                    to_node=edge_data["to"],
                )


# Insert the data from JSON to Neo4j
insert_data_into_neo4j(cfg_data)

# Close the Neo4j connection
driver.close()

print("Data has been successfully inserted into Neo4j.")
