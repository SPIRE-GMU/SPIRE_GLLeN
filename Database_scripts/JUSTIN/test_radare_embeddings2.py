#!/usr/bin/env python3
import os
import re
from collections import defaultdict
from neo4j import GraphDatabase
import networkx as nx

# === Neo4j Connection Details ===
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "rootboot"

# === Parsing Function ===
def parse_radare_structure(dot_content):
    """
    Parse a radare .dot file (from dot_data_radare) to extract structural features:
      - num_blocks: number of unique nodes (basic blocks) that appear in any edge definition.
      - num_edges: total number of edges.
      - num_loops: loops detected via strongly connected component analysis.
      - num_decisions: count of nodes with two or more outgoing edges.
    
    Assumes that nodes are denoted as hexadecimal strings (e.g. "0x400abc").
    """
    # Replace literal newline markers with actual newlines.
    dot_content = dot_content.replace("\\l", "\n").replace("\\n", "\n")
    lines = dot_content.splitlines()
    
    nodes = set()
    edges = []
    outgoing = defaultdict(list)
    
    # Regular expression to match edge definitions of the form: "0x..." -> "0x..."
    edge_re = re.compile(r'"(0x[0-9a-fA-F]+)"\s*->\s*"(0x[0-9a-fA-F]+)"')
    for line in lines:
        line = line.strip()
        if "->" in line:
            for match in edge_re.finditer(line):
                src, dst = match.group(1), match.group(2)
                nodes.add(src)
                nodes.add(dst)
                edges.append((src, dst))
                outgoing[src].append(dst)
    
    # Count decision nodes: nodes with two or more outgoing edges.
    decision_count = sum(1 for targets in outgoing.values() if len(targets) >= 2)
    
    # Build a directed graph to calculate loops using strongly connected components.
    G = nx.DiGraph()
    G.add_edges_from(edges)
    loops = 0
    for component in nx.strongly_connected_components(G):
        if len(component) > 1:
            loops += 1
        elif len(component) == 1:
            node = next(iter(component))
            if G.has_edge(node, node):
                loops += 1

    return {
        "num_blocks": len(nodes),
        "num_edges": len(edges),
        "num_loops": loops,
        "num_decisions": decision_count
    }

# === Update the Function Node in Neo4j ===
def update_function_with_embeddings(driver, unique_id, stats):
    """
    Update the Function node (identified by unique_id) with the computed statistics.
    New properties stored are:
      - node_count_radare
      - edge_count_radare
      - loop_count_radare
      - decision_count_radare
    """
    with driver.session() as session:
        session.run(
            """
            MATCH (f:Function {unique_id: $unique_id})
            SET f.node_count_radare = $node_count,
                f.edge_count_radare = $edge_count,
                f.loop_count_radare = $loop_count,
                f.decision_count_radare = $decision_count
            """,
            unique_id=unique_id,
            node_count=stats["num_blocks"],
            edge_count=stats["num_edges"],
            loop_count=stats["num_loops"],
            decision_count=stats["num_decisions"]
        )

# === Retrieve Function Nodes with dot_data_radare Property ===
def get_function_nodes_with_dot_radare(driver):
    """
    Retrieve all Function nodes that have a non-null, non-empty dot_data_radare property.
    Returns a list of dictionaries with each node's unique_id and dot_data_radare.
    """
    query = """
    MATCH (f:Function)
    WHERE f.dot_data_radare IS NOT NULL AND f.dot_data_radare <> ''
    RETURN f.unique_id AS unique_id, f.dot_data_radare AS dot_data_radare
    """
    results = []
    with driver.session() as session:
        for record in session.run(query):
            results.append({
                "unique_id": record["unique_id"],
                "dot_data_radare": record["dot_data_radare"]
            })
    return results

# === Verify the Updates by Re-Querying Each Node ===
def verify_updates(driver, unique_id):
    """
    Retrieve and print the updated radare embedding properties for the Function node with the given unique_id.
    """
    query = """
    MATCH (f:Function {unique_id: $unique_id})
    RETURN f.node_count_radare AS node_count,
           f.edge_count_radare AS edge_count,
           f.loop_count_radare AS loop_count,
           f.decision_count_radare AS decision_count
    """
    with driver.session() as session:
        record = session.run(query, unique_id=unique_id).single()
        if record:
            print(f"Function {unique_id} updated embeddings:")
            print(f"  Node Count: {record['node_count']}")
            print(f"  Edge Count: {record['edge_count']}")
            print(f"  Loop Count: {record['loop_count']}")
            print(f"  Decision Count: {record['decision_count']}")
        else:
            print(f"Function {unique_id} not found in the database.")

# === Main Execution ===
def main():
    # Connect to Neo4j.
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    # Retrieve all Function nodes with dot_data_radare.
    functions = get_function_nodes_with_dot_radare(driver)
    if not functions:
        print("No Function nodes with dot_data_radare found in Neo4j.")
        driver.close()
        return
    
    print(f"Found {len(functions)} function(s) with dot_data_radare.\n")
    
    # Process each function: parse, compute embeddings, update node, then verify.
    for func in functions:
        unique_id = func["unique_id"]
        dot_data = func["dot_data_radare"]
        print(f"Processing Function {unique_id}...")
        try:
            stats = parse_radare_structure(dot_data)
            print("Computed Embeddings:")
            print(f"  Basic Blocks: {stats['num_blocks']}")
            print(f"  Edges: {stats['num_edges']}")
            print(f"  Loops: {stats['num_loops']}")
            print(f"  Decision Points: {stats['num_decisions']}")
            
            update_function_with_embeddings(driver, unique_id, stats)
            verify_updates(driver, unique_id)
        except Exception as e:
            print(f"Error processing function {unique_id}: {e}")
        print("-" * 50)
    
    driver.close()
    print("All functions processed and embeddings updated.")

if __name__ == "__main__":
    main()

