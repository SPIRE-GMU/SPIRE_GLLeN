#!/usr/bin/env python3
"""
Compare two CFG .dot files (Radare vs. GCC) with structure‑only
Graph‑Edit‑Distance and rich diagnostics.

* Removes GCC ENTRY/EXIT (basic_block_0 / 1) automatically.
* Optional smart‑expand of Radare’s final branch→RET helper with
  --smart radare
* -d / --debug  prints full node / edge lists and extra‑node diffs.

Author: 2024‑04‑10
"""
import sys, re, argparse, textwrap
import networkx as nx, pydot
from collections import defaultdict


# ───────────────────────── .dot → networkx helpers ────────────
def flatten_pydot(dot: str) -> pydot.Dot:
    graphs = pydot.graph_from_dot_data(dot)
    if not graphs:
        raise ValueError("empty dot")
    root = pydot.Dot(graph_type="digraph")

    def rec(g):
        for n in g.get_nodes():
            if n.get_name() not in {"node", "graph", "edge"}:
                root.add_node(n)
        for e in g.get_edges():
            root.add_edge(e)
        for sg in g.get_subgraphs():
            rec(sg)

    for g in graphs:
        rec(g)
    return root


def load_dot(path: str) -> nx.DiGraph:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        data = f.read()
    G = nx.nx_pydot.from_pydot(flatten_pydot(data))
    H = nx.DiGraph()
    for n in G:
        H.add_node(n.split(":")[0])
    for u, v in G.edges():
        H.add_edge(u.split(":")[0], v.split(":")[0])
    return H


# ───────────────────────── structural stats ───────────────────
def stats(G):
    loops = sum(
        1
        for c in nx.strongly_connected_components(G)
        if len(c) > 1 or (len(c) == 1 and G.has_edge(*([next(iter(c))] * 2)))
    )
    dec = sum(1 for n in G if G.out_degree(n) >= 2)
    return dict(V=G.number_of_nodes(), E=G.number_of_edges(), loops=loops, dec=dec)


# ───────────────────────── GCC clean‑up ───────────────────────
def gcc_cleanup(G):
    H = G.copy()
    drop = [n for n in H if n.endswith("basic_block_0") or n.endswith("basic_block_1")]
    H.remove_nodes_from(drop)
    return H


# ───────────────────────── Radare smart‑expand ───────────────
def radare_expand(G, dbg=False):
    H = G.copy()
    added = []
    sid = 0
    for u, v in list(H.edges()):
        if H.out_degree(u) >= 2 and H.out_degree(v) == 0:
            s = f"SYNTH_{sid}"
            sid += 1
            H.remove_edge(u, v)
            H.add_node(s)
            H.add_edge(u, s)
            H.add_edge(s, v)
            added.append(s)
    if dbg and added:
        print("[smart‑expand] inserted:", added)
    return H


# ───────────────────────── pretty dump ────────────────────────
def dump(G, name):
    print(f"[{name}] Nodes ({G.number_of_nodes()}):")
    for n in sorted(G):
        print(f"    {n}")
    print(f"[{name}] Edges ({G.number_of_edges()}):")
    for u, v in sorted(G.edges()):
        print(f"    {u} -> {v}")
    print(f"[{name}] Loops detected: {stats(G)['loops']}\n")


# ───────────────────────── CLI ────────────────────────────────
ap = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=textwrap.dedent(__doc__),
)
ap.add_argument("radare_dot")
ap.add_argument("gcc_dot")
ap.add_argument(
    "--smart",
    choices=["radare", "none"],
    default="none",
    help="apply extra expansion to Radare graph",
)
ap.add_argument("-d", "--debug", action="store_true")
args = ap.parse_args()

# ───────────────────────── load & preprocess ──────────────────
G1 = load_dot(args.radare_dot)  # Radare
G2 = load_dot(args.gcc_dot)  # GCC original
G2 = gcc_cleanup(G2)  # drop bb0/1 always

if args.smart == "radare":
    G1 = radare_expand(G1, dbg=args.debug)

# ───────────────────────── debug dump ─────────────────────────
if args.debug:
    dump(G1, "Radare")
    dump(G2, "GCC")

s1, s2 = stats(G1), stats(G2)
print(
    f"{args.radare_dot.split('/')[-1]}: |V|={s1['V']} |E|={s1['E']} loops={s1['loops']} decisions={s1['dec']}"
)
print(
    f"{args.gcc_dot.split('/')[-1]}: |V|={s2['V']} |E|={s2['E']} loops={s2['loops']} decisions={s2['dec']}\n"
)

ged = nx.graph_edit_distance(G1, G2, timeout=5)
print(f"Structural GED = {ged if ged is not None else '∞':.2f}")

if args.debug:
    iso = nx.is_isomorphic(G1, G2)
    print(f"[iso] isomorphic? {iso}")
    print(f"\n[diff] extra nodes Radare→ {[n for n in G1 if n not in G2]}")
    print(f"[diff] extra nodes GCC   → {[n for n in G2 if n not in G1]}")
