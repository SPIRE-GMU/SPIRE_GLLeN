#!/usr/bin/env python3
import argparse
import time
import pydot
import networkx as nx
from networkx.algorithms import isomorphism
from neo4j import GraphDatabase

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "rootboot"  # Update if needed

# Overhead nodes that we remove from both graphs
OVERHEAD_NODES = ["fn_0_basic_block_0", "fn_0_basic_block_1"]

# Bound thresholds (tweak as needed)
MAX_NODE_DIFF = 1       # If node counts differ by more than this, skip GED
MAX_EDGE_DIFF = 1       # If edge counts differ by more than this, skip GED
GED_UPPER_BOUND = 15     # If the edit distance cost goes above this, return None

###############################################################################
# Helpers to parse/flatten .dot
###############################################################################
def flatten_pydot_graph_string(dot_data):
    graphs = pydot.graph_from_dot_data(dot_data)
    if not graphs:
        raise ValueError("No graphs found in the provided .dot data string.")
    combined = pydot.Dot(graph_type='digraph')

    def add_from_subgraph(pg):
        for node in pg.get_nodes():
            if node.get_name() not in ["node", "graph", "edge"]:
                combined.add_node(node)
        for edge in pg.get_edges():
            combined.add_edge(edge)
        for sub in pg.get_subgraphs():
            add_from_subgraph(sub)

    for g in graphs:
        add_from_subgraph(g)
    return combined

def read_and_normalize_dot_string(dot_data):
    pydot_combined = flatten_pydot_graph_string(dot_data)
    G = nx.nx_pydot.from_pydot(pydot_combined)
    return normalize_graph(G)

def flatten_pydot_graph_file(dot_file):
    graphs = pydot.graph_from_dot_file(dot_file)
    if not graphs:
        raise ValueError(f"No graphs found in {dot_file}")
    combined = pydot.Dot(graph_type='digraph')

    def add_from_subgraph(pg):
        for node in pg.get_nodes():
            if node.get_name() not in ["node", "graph", "edge"]:
                combined.add_node(node)
        for edge in pg.get_edges():
            combined.add_edge(edge)
        for sub in pg.get_subgraphs():
            add_from_subgraph(sub)

    for g in graphs:
        add_from_subgraph(g)
    return combined

def read_and_normalize_dot_file(dot_file):
    pydot_combined = flatten_pydot_graph_file(dot_file)
    G = nx.nx_pydot.from_pydot(pydot_combined)
    return normalize_graph(G)

def normalize_graph(G):
    """
    Remove port suffixes (like :s or :n) from node IDs.
    """
    newG = nx.DiGraph()
    mapping = {}
    for node in G.nodes():
        base_id = node.split(":")[0]
        mapping[node] = base_id
        newG.add_node(base_id)
    for (u, v) in G.edges():
        new_u = mapping[u]
        new_v = mapping[v]
        newG.add_edge(new_u, new_v)
    return newG

def filter_overhead_nodes(G, overhead_nodes):
    filtered = G.copy()
    for node in overhead_nodes:
        if node in filtered:
            filtered.remove_node(node)
    return filtered

###############################################################################
# compare_graphs with bounding
###############################################################################
def compare_graphs(func_name, g_r2, g_gcc):
    """
    Returns (is_isomorphic(bool), ged(float)), 
    but uses:
      - a node/edge count check to skip huge differences
      - an upper_bound=GED_UPPER_BOUND in graph_edit_distance
    We print debug logs to show timing and any skipping.
    """
    # 1) Quick size check
    diff_nodes = abs(g_r2.number_of_nodes() - g_gcc.number_of_nodes())
    diff_edges = abs(g_r2.number_of_edges() - g_gcc.number_of_edges())
    if diff_nodes > MAX_NODE_DIFF or diff_edges > MAX_EDGE_DIFF:
        print(f"[DEBUG] {func_name}: Node/edge count differs too much (Ndiff={diff_nodes}, Ediff={diff_edges}). Skipping GED.")
        # We can still do isomorphism if you like, or skip it too. 
        # For now, let's do isomorphism, but likely it's not isomorphic:
        start_iso = time.time()
        GM = isomorphism.DiGraphMatcher(g_r2, g_gcc, node_match=lambda n1,n2: True)
        is_iso = GM.is_isomorphic()
        iso_elapsed = time.time() - start_iso
        print(f"[DEBUG] {func_name}: is_isomorphic() done in {iso_elapsed:.2f}s => {is_iso}")
        # Mark a big distance
        return is_iso, 9999.0

    # 2) isomorphism check
    print(f"[DEBUG] Checking isomorphism for {func_name}...")
    start_iso = time.time()
    GM = isomorphism.DiGraphMatcher(g_r2, g_gcc, node_match=lambda n1, n2: True)
    is_iso = GM.is_isomorphic()
    iso_elapsed = time.time() - start_iso
    print(f"[DEBUG] {func_name}: is_isomorphic() finished in {iso_elapsed:.2f}s (is_iso={is_iso})")

    # 3) Graph edit distance with an upper bound
    print(f"[DEBUG] Calculating GED for {func_name} with upper_bound={GED_UPPER_BOUND} ...")
    start_ged = time.time()
    ged_val = nx.graph_edit_distance(
        g_r2, g_gcc,
        node_match=lambda n1, n2: True,
        upper_bound=GED_UPPER_BOUND
    )
    ged_elapsed = time.time() - start_ged

    # If the computed value is None, that means it exceeded the upper bound
    if ged_val is None:
        ged_val = 9999.0
        print(f"[DEBUG] {func_name}: GED cost exceeded {GED_UPPER_BOUND}, returning 9999.")
    else:
        print(f"[DEBUG] {func_name}: GED = {ged_val:.2f} in {ged_elapsed:.2f}s")

    return is_iso, ged_val

###############################################################################
# main script
###############################################################################
def main():
    parser = argparse.ArgumentParser(description="Compare radare2 .dot with GCC-based .dot in Neo4j, using bounding strategies.")
    parser.add_argument("radare2_dot", help="Path to the radare2 .dot file")
    args = parser.parse_args()

    # 1) Read & normalize the radare2 file
    r2_norm = read_and_normalize_dot_file(args.radare2_dot)
    r2_filtered = filter_overhead_nodes(r2_norm, OVERHEAD_NODES)
    print(f"r2 CFG => {r2_filtered.number_of_nodes()} nodes, {r2_filtered.number_of_edges()} edges\n")

    # 2) Connect to Neo4j
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    session = driver.session()

    # 3) Get all function nodes with dot_data from DB, convert to list
    results = list(session.run("""
        MATCH (f:Function)
        WHERE f.dot_data IS NOT NULL
        RETURN f.name AS func_name, f.dot_data AS dot_data
    """))
    session.close()
    driver.close()

    total_funcs = len(results)
    if total_funcs == 0:
        print("No .dot data found in the database. Exiting.")
        return
    print(f"Found {total_funcs} function(s) in the DB. Starting comparisons...\n")

    start_time = time.time()
    all_results = []
    best_ged = float("inf")
    best_func = None

    # 4) Compare each function
    for i, record in enumerate(results, start=1):
        func_name = record["func_name"]
        gcc_dot_data = record["dot_data"]
        if not gcc_dot_data:
            continue

        print(f"\n--- Comparing {i}/{total_funcs}: function '{func_name}' ---")
        try:
            gcc_norm = read_and_normalize_dot_string(gcc_dot_data)
            gcc_filtered = filter_overhead_nodes(gcc_norm, OVERHEAD_NODES)

            is_iso, ged_val = compare_graphs(func_name, r2_filtered, gcc_filtered)
            all_results.append((func_name, is_iso, ged_val))

            # track best
            if ged_val < best_ged:
                best_ged = ged_val
                best_func = func_name
                print(f"[DEBUG] New best GED = {best_ged:.2f} (func={best_func})")
        except Exception as e:
            print(f"[Error in {func_name}] => {e}")

        # occasional progress
        if i % 10 == 0 or i == total_funcs:
            elapsed = time.time() - start_time
            left = total_funcs - i
            print(f"[PROGRESS] Processed {i}/{total_funcs} in {elapsed:.2f}s; {left} remain.")
            if best_func is not None:
                print(f"[PROGRESS] Current best => {best_func} (GED={best_ged:.2f})")

    total_elapsed = time.time() - start_time
    print(f"\nDone. Processed {total_funcs} functions in {total_elapsed:.2f} seconds.\n")

    if not all_results:
        print("No valid comparison results.")
        return

    # 5) Sort by ascending GED
    all_results.sort(key=lambda x: x[2])

    # 6) Show top 3
    print("Top 3 matches (by smallest Graph Edit Distance):")
    print(f"{'Rank':4s} {'FunctionName':30s} {'IsIso':6s} {'GED'}")
    for rank, (fname, iso, ged_val) in enumerate(all_results[:3], start=1):
        print(f"{rank:4d} {fname:30s} {str(iso):6s} {ged_val:.2f}")

    best_func_name, best_iso, best_ged_val = all_results[0]
    print(f"\nBest match => {best_func_name} (GED={best_ged_val:.2f}, IsIso={best_iso})")

if __name__ == "__main__":
    main()
