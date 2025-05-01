#!/usr/bin/env python3
import sys, os, re, math, time, logging
import networkx as nx, pydot
from collections import defaultdict
from neo4j import GraphDatabase
from networkx.algorithms.similarity import optimize_graph_edit_distance

# Suppress low-level Neo4j warnings
logging.getLogger("neo4j").setLevel(logging.ERROR)

# Neo4j config
NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD = ("bolt://localhost:7687", "neo4j", "rootboot")

# Early exit config
GED_THRESHOLD = 5
TIME_SLOWDOWN_FACTOR = 3.0


# ─────────────────────────── DOT helpers
def flatten_dot(txt: str) -> pydot.Dot:
    gs = pydot.graph_from_dot_data(txt)
    if not gs:
        raise ValueError("No graphs found in dot data")
    combo = pydot.Dot(graph_type="digraph")

    def add(pg):
        for n in pg.get_nodes():
            if n.get_name() not in {"node", "edge", "graph"}:
                combo.add_node(n)
        for e in pg.get_edges():
            combo.add_edge(e)
        for sg in pg.get_subgraphs():
            add(sg)

    for g in gs:
        add(g)
    return combo


def to_nx(txt: str) -> nx.DiGraph:
    G = nx.nx_pydot.from_pydot(flatten_dot(txt))
    H = nx.DiGraph()
    for n in G:
        H.add_node(n.split(":")[0])
    for u, v in G.edges():
        H.add_edge(u.split(":")[0], v.split(":")[0])
    return H


def clean_query(txt: str) -> nx.DiGraph:
    return to_nx(txt)


def stats(G: nx.DiGraph):
    loops = sum(
        1
        for c in nx.strongly_connected_components(G)
        if len(c) > 1 or (len(c) == 1 and G.has_edge(next(iter(c)), next(iter(c))))
    )
    decisions = sum(1 for n in G if G.out_degree(n) >= 2)
    return (G.number_of_nodes(), G.number_of_edges(), loops, decisions)


def euclid(v1, v2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))


def fetch_candidates():
    q = """
    MATCH (f:Function)
    WHERE f.dot_data_radare IS NOT NULL
      AND f.node_count IS NOT NULL
      AND f.edge_count IS NOT NULL
      AND f.loop_count IS NOT NULL
      AND f.decision_count IS NOT NULL
    RETURN f.unique_id AS uid, 
           f.name AS fname, 
           f.dot_data_radare AS dot_data_radare,
           f.node_count AS node_count,
           f.edge_count AS edge_count,
           f.loop_count AS loop_count,
           f.decision_count AS decision_count
    """
    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
        with driver.session() as s:
            return [dict(r) for r in s.run(q)]


# ─────────────────────────── Main logic
def main():
    if len(sys.argv) < 2:
        print("Usage: python3 GED_search.py <query.dot> [k_pass1] [top_n]")
        sys.exit(1)

    query_file = sys.argv[1]
    k_pass1 = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    top_n = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    try:
        with open(query_file) as f:
            q_text = f.read()
        G_q = clean_query(q_text)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    q_vec = stats(G_q)
    print("Query embedding (structural stats):", q_vec)

    cands = fetch_candidates()
    if not cands:
        print("No candidates found.")
        sys.exit(1)

    for c in cands:
        try:
            c["emb"] = tuple(
                map(
                    float,
                    (
                        c["node_count"],
                        c["edge_count"],
                        c["loop_count"],
                        c["decision_count"],
                    ),
                )
            )
            c["dist"] = euclid(q_vec, c["emb"])
        except:
            c["dist"] = float("inf")

    exact_matches = [c for c in cands if c["dist"] == 0.0]
    fuzzy_matches = sorted([c for c in cands if c["dist"] > 0], key=lambda x: x["dist"])
    total_exact = len(exact_matches)

    print(f"\nFound {total_exact} exact structural matches (Pass 1).")

    if total_exact >= k_pass1:
        rough = exact_matches
    else:
        needed = k_pass1 - total_exact
        rough = exact_matches + fuzzy_matches[:needed]

    print(f"\nPass 1 – Selected {len(rough)} candidates for GED refinement:")
    for i, c in enumerate(rough, 1):
        print(f"{i:2}. {c['fname']:<30} dist={c['dist']:.2f} emb={c['emb']}")

    # ─────────── GED hybrid refinement
    print("\nPass 2 – GED refinement (hybrid early exit)")
    best_ged = float("inf")
    best_time = float("inf")

    for c in rough:
        try:
            G_c = clean_query(c["dot_data_radare"])
            start = time.time()
            gen = optimize_graph_edit_distance(G_q, G_c)

            min_ged = float("inf")
            for g in gen:
                if g is None:
                    continue
                if g < min_ged:
                    min_ged = g
                if min_ged > best_ged + GED_THRESHOLD:
                    break
                if time.time() - start > best_time * TIME_SLOWDOWN_FACTOR:
                    break

            elapsed = time.time() - start
            c["ged"] = min_ged

            if c["ged"] < best_ged:
                best_ged = c["ged"]
            if elapsed < best_time:
                best_time = elapsed

            print(f"   {c['uid']} GED={c['ged']:.2f} in {elapsed:.2f}s")

        except Exception as e:
            c["ged"] = float("inf")
            print(f"   !! {c.get('uid')} GED error: {e}")

    # ─────────── Top N selection (Pass 2 override)
    rough.sort(key=lambda x: x["ged"])
    ged_zero = [c for c in rough if c.get("ged") == 0.0]
    if len(ged_zero) >= top_n:
        top = ged_zero
        print(f"\nFound {len(top)} exact GED matches (GED = 0). Returning all of them.")
    else:
        needed = top_n - len(ged_zero)
        others = [c for c in rough if c.get("ged") != 0.0]
        top = ged_zero + others[:needed]
        print(
            f"\nReturning {len(top)} total candidates after GED refinement (including {len(ged_zero)} exact matches)."
        )

    print(f"\nTop {len(top)} candidates after GED:")
    for i, c in enumerate(top, 1):
        print(f"{i:2}. {c['fname']:<30} GED={c['ged']:.2f} d1={c['dist']:.2f}")

    # ─────────── Retrieve C code
    snippets = []
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session() as s:
            for c in top:
                rec = s.run(
                    "MATCH (f:Function {unique_id:$u}) RETURN f.c_data AS c", u=c["uid"]
                ).single()
                snippets.append(rec["c"] if rec and rec["c"] else "[No c_data found]")
    finally:
        driver.close()

    print("\n───── C‑code snippets (best matches) ─────")
    for i, code in enumerate(snippets, 1):
        print(f"\n### Candidate {i}\n{code}")

    return snippets


if __name__ == "__main__":
    main()
