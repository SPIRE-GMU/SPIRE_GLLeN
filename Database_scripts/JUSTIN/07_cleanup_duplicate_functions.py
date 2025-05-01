#!/usr/bin/env python3
"""
07_cleanup_duplicate_functions.py  (REV‑4)
=========================================
**Purpose**  Permanently remove every `:Function` node whose `unique_id`
contains a radare tag (`_sym.`, `_fcn.`, `_sub.`).  We *do not* rename any of
these nodes; they are all deleted.  If a canonical node already exists, we first
copy `dot_data_radare` into it (if the canonical property is null) so that no
information is lost.

After running this script, you can safely re‑run
`05_radare_dot_to_neo4j.py` (REV‑2) to repopulate the correct nodes.

Dry‑run support
---------------
Pass `--dry-run` to preview what would be deleted or merged.

Examples
--------
```bash
python3 07_cleanup_duplicate_functions.py --dry-run -v   # preview only
python3 07_cleanup_duplicate_functions.py                # execute deletion
```
"""

import argparse
import logging
import os
import re

from neo4j import GraphDatabase, basic_auth

###############################################################################
# Configuration
###############################################################################
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "rootboot")

TAG_RE = re.compile(r"_(sym|fcn|sub)\..*$")

###############################################################################
# Helper
###############################################################################


def strip_tag(uid: str) -> str:
    """Return UID with trailing radare tag removed."""
    return TAG_RE.sub("", uid)


###############################################################################
# Main
###############################################################################


def main():
    ap = argparse.ArgumentParser(
        description="Delete radare‑prefixed duplicate Function nodes"
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="Preview actions without mutating the DB"
    )
    ap.add_argument("-v", action="count", default=0)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.v else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    driver = GraphDatabase.driver(
        NEO4J_URI, auth=basic_auth(NEO4J_USER, NEO4J_PASSWORD)
    )

    deleted = merged = 0

    with driver.session() as s:
        dupes = s.run(
            """
            MATCH (f:Function)
            WHERE f.unique_id CONTAINS '_sym.' OR f.unique_id CONTAINS '_fcn.' OR f.unique_id CONTAINS '_sub.'
            RETURN f.unique_id AS uid, f.dot_data_radare AS dot
            """
        ).data()

        logging.info("Found %d duplicate nodes to remove", len(dupes))

        for rec in dupes:
            dup_uid = rec["uid"]
            dup_dot = rec["dot"]
            base_uid = strip_tag(dup_uid)

            # Does canonical node exist?
            can_exists = s.run(
                "MATCH (f:Function {unique_id:$u}) RETURN count(f) AS n", u=base_uid
            ).single()["n"]

            if can_exists and dup_dot:
                merged += 1
                if args.dry_run:
                    logging.info(
                        "[dry] copy dot_data_radare to %s then DELETE %s",
                        base_uid,
                        dup_uid,
                    )
                else:
                    s.run(
                        """
                        MATCH (dup:Function {unique_id:$dup_uid})
                        MATCH (can:Function {unique_id:$can_uid})
                        WHERE can.dot_data_radare IS NULL
                        SET can.dot_data_radare = dup.dot_data_radare
                        DETACH DELETE dup
                        """,
                        dup_uid=dup_uid,
                        can_uid=base_uid,
                    )
                    deleted += 1
            else:
                # No canonical node or nothing to merge – just delete
                if args.dry_run:
                    logging.info("[dry] DELETE %s", dup_uid)
                else:
                    s.run(
                        "MATCH (d:Function {unique_id:$u}) DETACH DELETE d", u=dup_uid
                    )
                    deleted += 1

    driver.close()
    logging.info(
        "Summary: deleted=%d, merged_into_existing=%d, dry=%s",
        deleted,
        merged,
        args.dry_run,
    )


if __name__ == "__main__":
    main()
