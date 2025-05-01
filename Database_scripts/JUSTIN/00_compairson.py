#!/usr/bin/env python3
import argparse
import networkx as nx
import pydot
import matplotlib.pyplot as plt
from networkx.algorithms import isomorphism

# Default file paths (adjust these as needed)
DEFAULT_GCC_DOT = "accept_i16.c.015t.cfg.dot"
DEFAULT_R2_DOT = "accept_i16.dot"


def flatten_pydot_graph(dot_file):
    graphs = pydot.graph_from_dot_file(dot_file)
    if not graphs:
        raise ValueError(f"No graphs found in {dot_file}")
    combined = pydot.Dot(graph_type="digraph")

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


def read_flattened_dot(file_path):
    pydot_combined = flatten_pydot_graph(file_path)
    G = nx.nx_pydot.from_pydot(pydot_combined)
    return G


def normalize_graph(G):
    newG = nx.DiGraph()
    mapping = {node: node.split(":")[0] for node in G.nodes()}
    for orig, canon in mapping.items():
        newG.add_node(canon)
    for u, v in G.edges():
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


def main():
    parser = argparse.ArgumentParser(
        description="Compare two CFG DOT files for structural similarity."
    )
    parser.add_argument(
        "--gcc",
        "-g",
        type=str,
        default=DEFAULT_GCC_DOT,
        help="Path to GCC DOT file (default: {})".format(DEFAULT_GCC_DOT),
    )
    parser.add_argument(
        "--r2",
        "-r",
        type=str,
        default=DEFAULT_R2_DOT,
        help="Path to radare2 DOT file (default: {})".format(DEFAULT_R2_DOT),
    )
    parser.add_argument(
        "--graph", "-c", action="store_true", help="Display graphs using matplotlib"
    )
    args = parser.parse_args()

    gcc_dot_file = args.gcc
    r2_dot_file = args.r2
    show_graphs = args.graph

    # Process GCC DOT: flatten, normalize, and filter out overhead nodes.
    gcc_flat = read_flattened_dot(gcc_dot_file)
    gcc_norm = normalize_graph(gcc_flat)
    overhead = ["fn_0_basic_block_0", "fn_0_basic_block_1"]
    gcc_filtered = filter_overhead_nodes(gcc_norm, overhead)

    # Process radare2 DOT.
    r2_pydot = pydot.graph_from_dot_file(r2_dot_file)
    if not r2_pydot:
        raise ValueError(f"Could not load any graph from {r2_dot_file}")
    r2_graph = nx.nx_pydot.from_pydot(r2_pydot[0])

    # Final outputs.
    print(
        "GCC CFG: {} nodes, {} edges".format(
            gcc_filtered.number_of_nodes(), gcc_filtered.number_of_edges()
        )
    )
    print(
        "radare2 CFG: {} nodes, {} edges".format(
            r2_graph.number_of_nodes(), r2_graph.number_of_edges()
        )
    )

    node_match = lambda n1, n2: True
    GM = isomorphism.DiGraphMatcher(gcc_filtered, r2_graph, node_match=node_match)
    print("Isomorphic:", "Yes" if GM.is_isomorphic() else "No")

    ged = nx.graph_edit_distance(gcc_filtered, r2_graph)
    print("Graph edit distance: {:.1f}".format(ged))

    if show_graphs:
        plt.figure(figsize=(14, 7))
        plt.subplot(121)
        plt.title("Filtered CFG from GCC")
        pos1 = nx.spring_layout(gcc_filtered, seed=42)
        nx.draw_networkx(
            gcc_filtered,
            pos=pos1,
            with_labels=True,
            node_color="lightblue",
            edge_color="gray",
            font_size=8,
        )
        plt.axis("off")
        plt.subplot(122)
        plt.title("CFG from radare2")
        pos2 = nx.spring_layout(r2_graph, seed=42)
        nx.draw_networkx(
            r2_graph,
            pos=pos2,
            with_labels=True,
            node_color="lightgreen",
            edge_color="gray",
            font_size=8,
        )
        plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
