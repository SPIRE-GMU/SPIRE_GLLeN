#!/usr/bin/env python3
"""
06_fetch_function_info.py
=========================
Fetch **all** properties stored on a single `:Function` node (and optionally its
`BasicBlock`s) from Neo4j, *plus* a one‑line summary of how many `Function`
nodes exist in the database.

Usage
-----
```bash
python3 06_fetch_function_info.py --id <unique_id>
python3 06_fetch_function_info.py --name main   # match by f.name
python3 06_fetch_function_info.py --id foo_bar --with-blocks -o foo.json -v
```

Environment variables `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` override the
hard‑coded defaults.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from neo4j import GraphDatabase, basic_auth

###############################################################################
# Configuration
###############################################################################
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "rootboot")

###############################################################################
# Helper – run Cypher and return records as list of dicts
###############################################################################


def run_cypher(session, query: str, **params):
    logging.debug("Cypher>>>\n%s\nparams=%s", query.strip(), params or "{}")
    try:
        return [rec.data() for rec in session.run(query, **params)]
    except Neo4jError as e:
        logging.error("Cypher error [%s]: %s", e.code, e.message)
        raise


###############################################################################
# Main
###############################################################################


def main():
    parser = argparse.ArgumentParser(
        description="Fetch a Function node from Neo4j and show DB stats"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--id", dest="uid", help="Function.unique_id to match")
    group.add_argument("--name", help="Function.name to match (may return >1)")
    parser.add_argument(
        "--with-blocks", action="store_true", help="Include BasicBlock list"
    )
    parser.add_argument(
        "-o", "--out", type=Path, help="Write JSON output to file instead of stdout"
    )
    parser.add_argument(
        "-v", action="count", default=0, help="Increase verbosity (‑v DEBUG)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.v else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    driver = GraphDatabase.driver(
        NEO4J_URI, auth=basic_auth(NEO4J_USER, NEO4J_PASSWORD)
    )

    with driver.session() as session:
        # --------------------------------------------------------------
        # 0. How many Function nodes in total?
        # --------------------------------------------------------------
        total_funcs = run_cypher(session, "MATCH (f:Function) RETURN count(f) AS n")[0][
            "n"
        ]
        logging.info("Total Function nodes in DB: %d", total_funcs)

        # --------------------------------------------------------------
        # 1. Fetch requested Function
        # --------------------------------------------------------------
        if args.uid:
            match_clause = "MATCH (f:Function {unique_id: $val})"
            params = {"val": args.uid}
        else:
            match_clause = "MATCH (f:Function {name: $val})"
            params = {"val": args.name}

        query = f"""
            {match_clause}
            RETURN properties(f) AS func_props
        """
        records = run_cypher(session, query, **params)
        if not records:
            logging.error("No Function node found for value='%s'", params["val"])
            sys.exit(1)

        func_props = records[0]["func_props"]
        result_obj = {
            "DB_total_functions": total_funcs,
            "Function": func_props,
        }

        # --------------------------------------------------------------
        # 2. Optionally fetch BasicBlocks
        # --------------------------------------------------------------
        if args.with_blocks:
            logging.info("Fetching BasicBlocks for this function …")
            block_query = f"""
                {match_clause}
                MATCH (f)-[:STARTS_AT|NEXT*0..]->(b:BasicBlock)
                WHERE b.function_unique_id = f.unique_id
                RETURN b.id AS id, properties(b) AS props
            """
            blocks = run_cypher(session, block_query, **params)
            blocks_sorted = sorted(blocks, key=lambda x: x["id"])
            result_obj["BasicBlocks"] = {
                rec["id"]: rec["props"] for rec in blocks_sorted
            }

    driver.close()

    # --------------------------------------------------------------
    # 3. Output JSON
    # --------------------------------------------------------------
    json_str = json.dumps(result_obj, indent=2, ensure_ascii=False)
    if args.out:
        args.out.write_text(json_str)
        logging.info("Written output to %s", args.out)
    else:
        print(json_str)


if __name__ == "__main__":
    main()
