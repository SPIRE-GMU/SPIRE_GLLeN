#!/usr/bin/env python3
"""
Script to process a DOT file, compute structural statistics, and update
the corresponding Function node in Neo4j with these values.

Usage:
    python3 update_stats.py <path/to/function.dot>

This script:
  - Reads the DOT file.
  - Converts the DOT content to a NetworkX graph (stripping any port suffixes).
  - Computes structural statistics:
       * Number of nodes
       * Number of edges
       * Number of loops (non-trivial strongly connected components or self-loops)
       * Number of decision points (nodes with at least two outgoing edges)
  - Uses the DOT file's basename (without extension) as the unique_id 
    to search for the Function node in Neo4j.
  - If found, updates its fields (node_count, edge_count, loop_count, decision_count).
  - Prints verbose output indicating the calculated numbers, whether the node was found,
    and whether the update was successful.
"""

import sys
import os
import re
import math
import networkx as nx
import pydot
from neo4j import GraphDatabase

# ───────────────────────────────────────── Neo4j connection settings
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "rootboot"


# ───────────────────────────────────────── DOT parsing helpers
def flatten_dot(txt: str) -> pydot.Dot:
    """
    Parse DOT data and combine graphs/subgraphs into one DOT graph.
    """
    graphs = pydot.graph_from_dot_data(txt)
    if not graphs:
        raise ValueError("No graphs found in dot data.")
    combo = pydot.Dot(graph_type="digraph")

    def add(pg):
        for n in pg.get_nodes():
            if n.get_name() not in {"node", "edge", "graph"}:
                combo.add_node(n)
        for e in pg.get_edges():
            combo.add_edge(e)
        for sg in pg.get_subgraphs():
            add(sg)

    for g in graphs:
        add(g)
    return combo


def to_nx(txt: str) -> nx.DiGraph:
    """
    Convert DOT data to a NetworkX DiGraph and strip any port suffixes (e.g., ':s', ':n').
    """
    G = nx.nx_pydot.from_pydot(flatten_dot(txt))
    H = nx.DiGraph()
    for n in G:
        H.add_node(n.split(":")[0])
    for u, v in G.edges():
        H.add_edge(u.split(":")[0], v.split(":")[0])
    return H


def clean_dot(txt: str) -> nx.DiGraph:
    """
    Process the DOT file text into a normalized NetworkX graph.
    (This does not apply any synthetic node expansion, matching our previous radare logic.)
    """
    return to_nx(txt)


# ───────────────────────────────────────── Structural statistics calculation
def stats(G: nx.DiGraph):
    """
    Compute structural statistics from the graph:
      - Number of nodes.
      - Number of edges.
      - Loop count: count each strongly connected component that has more than one node
        or a single node with a self-loop.
      - Decision count: count nodes with two or more outgoing edges.
    Returns a tuple: (num_nodes, num_edges, loop_count, decision_count)
    """
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    loops = sum(
        1
        for comp in nx.strongly_connected_components(G)
        if len(comp) > 1
        or (len(comp) == 1 and G.has_edge(next(iter(comp)), next(iter(comp))))
    )
    decisions = sum(1 for n in G if G.out_degree(n) >= 2)
    return (num_nodes, num_edges, loops, decisions)


# ───────────────────────────────────────── Neo4j update helper
def update_function_stats(
    unique_id, node_count, edge_count, loop_count, decision_count
):
    """
    Update the Function node in Neo4j (matched by unique_id) with the computed statistics.
    Prints verbose output for debugging and confirmation.
    """
    query = """
    MATCH (f:Function {unique_id: $uid})
    SET f.node_count = $node_count,
        f.edge_count = $edge_count,
        f.loop_count = $loop_count,
        f.decision_count = $decision_count
    RETURN f.name AS fname
    """
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            result = session.run(
                query,
                uid=unique_id,
                node_count=node_count,
                edge_count=edge_count,
                loop_count=loop_count,
                decision_count=decision_count,
            )
            record = result.single()
            if record:
                fname = record["fname"]
                print(
                    f"\nFunction '{fname}' (unique_id: {unique_id}) updated successfully:"
                )
                print(f"  Number of Nodes     : {node_count}")
                print(f"  Number of Edges     : {edge_count}")
                print(f"  Number of Loops     : {loop_count}")
                print(f"  Number of Decisions : {decision_count}")
            else:
                print(
                    f"\nNo Function node with unique_id '{unique_id}' was found in Neo4j. (Update skipped)"
                )
        driver.close()
    except Exception as e:
        print(f"Error updating function stats in Neo4j: {e}")


# ───────────────────────────────────────── Main routine
def main():
    if len(sys.argv) < 2:
        print("Usage: python3 update_stats.py <path/to/function.dot>")
        sys.exit(1)

    dot_file = sys.argv[1]
    if not os.path.isfile(dot_file):
        print(f"DOT file not found: {dot_file}")
        sys.exit(1)

    # Use the basename (without extension) as the unique_id.
    unique_id = os.path.splitext(os.path.basename(dot_file))[0]
    print(f"Processing DOT file for function '{unique_id}': {dot_file}")

    try:
        with open(dot_file, "r") as f:
            dot_text = f.read()
    except Exception as e:
        print(f"Error reading DOT file: {e}")
        sys.exit(1)

    try:
        graph = clean_dot(dot_text)
    except Exception as e:
        print(f"Error processing DOT file: {e}")
        sys.exit(1)

    node_count, edge_count, loop_count, decision_count = stats(graph)

    print("\nCalculated statistics from DOT file:")
    print(f"  Number of Nodes     : {node_count}")
    print(f"  Number of Edges     : {edge_count}")
    print(f"  Number of Loops     : {loop_count}")
    print(f"  Number of Decisions : {decision_count}\n")

    update_function_stats(unique_id, node_count, edge_count, loop_count, decision_count)


if __name__ == "__main__":
    main()
