#!/usr/bin/env python3
import os
import re
import json

###############################################################################
# Regex Patterns
###############################################################################
ATTR_PAIR_REGEX = re.compile(r'(\w+)\s*=\s*("([^"]*)"|[^",\s]+)')
NODE_LINE_REGEX = re.compile(r"^(\S+)\s*\[(.*?)\];\s*$")
EDGE_LINE_REGEX = re.compile(r"^(\S+)\s*->\s*(\S+)\s*\[(.*?)\];\s*$")
PORT_SUFFIX_REGEX = re.compile(r":[a-zA-Z]+$")
LABEL_LINE_REGEX = re.compile(r'^label="([^"]*)"\s*;')


###############################################################################
# Helpers
###############################################################################
def parse_attributes(attr_str):
    attrs = {}
    for match in ATTR_PAIR_REGEX.finditer(attr_str):
        key = match.group(1)
        raw_val = match.group(2)
        val = (
            raw_val[1:-1]
            if raw_val.startswith('"') and raw_val.endswith('"')
            else raw_val
        )
        attrs[key] = val
    return attrs


def parse_dot_file(dot_path):
    with open(dot_path, "r", encoding="utf-8", errors="replace") as f:
        lines = [l.strip() for l in f]

    function_name = None
    nodes = []
    edges = []
    fallback_name = os.path.splitext(os.path.basename(dot_path))[0]

    # Extract function name
    for line in lines:
        label_match = LABEL_LINE_REGEX.match(line)
        if label_match:
            raw_lbl = label_match.group(1).strip()
            function_name = raw_lbl[:-2].strip() if raw_lbl.endswith("()") else raw_lbl
            break
    if not function_name:
        function_name = fallback_name

    seen_nodes = set()
    edge_node_ids = set()

    for line in lines:
        node_m = NODE_LINE_REGEX.match(line)
        if node_m:
            node_id_raw = node_m.group(1)
            attr_part = node_m.group(2)
            node_id = PORT_SUFFIX_REGEX.sub("", node_id_raw)
            attrs = parse_attributes(attr_part)
            nodes.append({"id": node_id, "attributes": attrs})
            seen_nodes.add(node_id)
            continue

        edge_m = EDGE_LINE_REGEX.match(line)
        if edge_m:
            from_id = PORT_SUFFIX_REGEX.sub("", edge_m.group(1))
            to_id = PORT_SUFFIX_REGEX.sub("", edge_m.group(2))
            attr_part = edge_m.group(3)
            attrs = parse_attributes(attr_part)
            edges.append({"from": from_id, "to": to_id, "attributes": attrs})
            edge_node_ids.update([from_id, to_id])
            continue

    # Add nodes seen in edges but never declared
    for missing_id in edge_node_ids - seen_nodes:
        nodes.append(
            {
                "id": missing_id,
                "attributes": {
                    "label": missing_id,
                    "style": "inferred",
                    "fillcolor": "gray",
                },
            }
        )

    return {"function_name": function_name, "nodes": nodes, "edges": edges}


def parse_and_save_dot(dot_path, out_json_path):
    parsed = parse_dot_file(dot_path)
    with open(out_json_path, "w", encoding="utf-8") as out_f:
        json.dump(parsed, out_f, indent=2)
    print(f"[OK] Parsed '{dot_path}' → '{out_json_path}'")


def main():
    dot_dir = "/home/spire2/SPIRE_GLLeN/Neo4J/justin/dot_files_newDB"
    json_dir = "/home/spire2/SPIRE_GLLeN/Neo4J/justin/json_files"
    os.makedirs(json_dir, exist_ok=True)

    for filename in os.listdir(dot_dir):
        if filename.endswith(".dot"):
            dot_path = os.path.join(dot_dir, filename)
            out_path = os.path.join(json_dir, os.path.splitext(filename)[0] + ".json")
            parse_and_save_dot(dot_path, out_path)


if __name__ == "__main__":
    main()
    print("✅ Done parsing all .dot files.")
