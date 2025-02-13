#!/usr/bin/env python3
import os
import sys
import re
import numpy as np
from neo4j import GraphDatabase
from sklearn.metrics.pairwise import cosine_similarity

###############################################################################
# Neo4j Connection
###############################################################################
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "rootboot"  # update as needed

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


###############################################################################
# 1) Parse a .cfg File, unifying block1 => end
###############################################################################
def parse_cfg_for_knn(cfg_path):
    """
    We parse the .cfg lines. Key points:
      - If the CFG references block1, we unify that with 'end'.
      - We add a function node 'func'.
      - If block2 exists, we add edge (func->2).
      - We do NOT automatically add a final-block->end edge (the raw edges do it).
      - loops_count is found via ';; X loops found'.

    In the end, we produce a set of nodes and edges that match your pipeline:
      nodes = { all blocks except block1, plus 'end' if block1 was referenced, plus 'func' }
      edges = raw 'succs' edges, except references to 1 become 'end', plus 'func->2' if block2 is present
    """
    with open(cfg_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    loops_count = 0
    blocks = []  # store integer block IDs (like 2, 3, etc.)
    block_contents = {}  # e.g. { 2: ["<bb 2>", "some code"] }

    edges = []  # list of {"from": X, "to": Y}, where X,Y are either blockID or 'end'
    block1_referenced = False

    current_bb = None
    current_lines = []

    # finalize block helper
    def finalize_block(bb_id, codeacc):
        block_contents[bb_id] = codeacc[:]

    for line in lines:
        line_stripped = line.strip()

        # detect loops
        m_loop = re.match(r";;\s*(\d+)\s+loops found", line_stripped)
        if m_loop:
            loops_count = int(m_loop.group(1))

        # detect <bb X>
        if line_stripped.startswith("<bb"):
            # finalize the previous
            if current_bb is not None:
                finalize_block(current_bb, current_lines)
            # start new
            m_bb = re.match(r"<bb (\d+)>", line_stripped)
            if m_bb:
                current_bb = int(m_bb.group(1))
                current_lines = [line_stripped]
                if current_bb not in blocks and current_bb != 1:
                    blocks.append(current_bb)
            else:
                current_bb = None
                current_lines = []
        elif "succs" in line_stripped:
            # e.g. ";; 2 succs { 3 7 }"
            m_succ = re.match(
                r";;\s*(\d+)\s+succs\s+\{\s*([\d\s]+)\s*\}", line_stripped
            )
            if m_succ:
                src = int(m_succ.group(1))
                targets_str = m_succ.group(2).strip()
                targets = [int(x) for x in targets_str.split() if x.isdigit()]
                for t in targets:
                    if t == 1:
                        # unify block1 with 'end'
                        edges.append({"from": src, "to": "end"})
                        block1_referenced = True
                    else:
                        edges.append({"from": src, "to": t})
                        if t not in blocks and t != 1:
                            blocks.append(t)
        else:
            # accumulate lines if in a block
            if current_bb is not None:
                current_lines.append(line_stripped)

    # finalize last block
    if current_bb is not None:
        finalize_block(current_bb, current_lines)

    # now we have blocks = [2, 3, etc.], block_contents = {2: [...], 3: [...]} if any
    # unify function node => "func"
    # unify end node => "end" only if block1 was referenced

    # we always add the function node
    # if block1_referenced, we add the 'end' node
    # and if block2 exists, add edge (func->2)

    # check if block2 exists
    block2_exists = 2 in blocks

    # new edge: (func->2) if block2 exists
    if block2_exists:
        edges.append({"from": "func", "to": 2})

    # build final node list
    #  - all the blocks in 'blocks'
    #  - plus 'func'
    #  - plus 'end' if block1 was referenced
    node_list = []
    for b in blocks:
        node_list.append({"block_id": b})
    node_list.append({"block_id": "func"})  # function node
    if block1_referenced:
        node_list.append({"block_id": "end"})

    return node_list, edges, loops_count


###############################################################################
# 2) Compute embedding
###############################################################################
def compute_embedding_knn(nodes, edges, loops_count):
    return [len(nodes), len(edges), loops_count]


###############################################################################
# 3) Load DB Embeddings
###############################################################################
from neo4j import GraphDatabase


def load_db_embeddings():
    with driver.session() as session:
        res = session.run(
            """
            MATCH (f:Function)
            RETURN f.cfg_id AS cfg_id, f.embedding AS embedding
        """
        )
        data = []
        for r in res:
            cid = r["cfg_id"]
            emb = r["embedding"]
            data.append((cid, emb))
        return data


###############################################################################
# 4) KNN
###############################################################################
from sklearn.metrics.pairwise import cosine_similarity


def knn_search(new_emb, db_embs, top_k=3, use_cosine=True):
    import math

    query_vec = np.array(new_emb).reshape(1, -1)
    all_ids, all_vecs = [], []
    for cid, emb in db_embs:
        all_ids.append(cid)
        all_vecs.append(emb)
    all_vecs = np.array(all_vecs)

    if use_cosine:
        sims = cosine_similarity(query_vec, all_vecs)[0]
        idx = np.argsort(-sims)
        best = idx[:top_k]
        return [(all_ids[i], sims[i]) for i in best]
    else:
        diff = all_vecs - query_vec
        dist = np.sqrt(np.sum(diff**2, axis=1))
        idx = np.argsort(dist)
        best = idx[:top_k]
        return [(all_ids[i], dist[i]) for i in best]


###############################################################################
# (Optional) Code
###############################################################################
def get_function_code(cfg_id):
    with driver.session() as session:
        rec = session.run(
            """
            MATCH (f:Function {cfg_id: $cid})
            RETURN f.c_data AS cdata, f.code AS fcode
        """,
            cid=cfg_id,
        ).single()
        if rec:
            return rec["cdata"], rec["fcode"]
        return None, None


###############################################################################
# MAIN
###############################################################################
def main():
    if len(sys.argv) < 2:
        print("Usage: python knn_search.py <new.cfg>")
        sys.exit(1)

    new_cfg = sys.argv[1]
    if not os.path.isfile(new_cfg):
        print(f"[ERROR] {new_cfg} not found.")
        sys.exit(1)

    # parse
    node_list, edge_list, loops = parse_cfg_for_knn(new_cfg)

    # embed
    new_emb = compute_embedding_knn(node_list, edge_list, loops)
    print(f"[INFO] new embedding = {new_emb}")

    # load
    db_embs = load_db_embeddings()
    if not db_embs:
        print("[WARN] DB empty or no embeddings found.")
        return

    # KNN
    topk = 3
    results = knn_search(new_emb, db_embs, top_k=topk, use_cosine=True)
    print(f"\n[INFO] top {topk} results by Cosine Similarity:\n")
    for cid, score in results:
        print(f" - {cid}, similarity={score:.4f}")
        # optionally fetch code
        cd, fc = get_function_code(cid)
        if cd:
            print(f"   c_data length={len(cd)}")
        if fc:
            print(f"   function code length={len(fc)}")


if __name__ == "__main__":
    main()
