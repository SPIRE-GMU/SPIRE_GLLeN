#!/usr/bin/env python3
import os
import re
import json
import sys

###############################################################################
# Regex Patterns
###############################################################################
# This pattern captures "key=value" pairs inside the bracket,
# like shape=Mdiamond, style=filled, label="stuff", etc.
# - We allow for key=value OR key="value" forms.
# - Values can be unquoted or in quotes.
ATTR_PAIR_REGEX = re.compile(r'(\w+)\s*=\s*("([^"]*)"|[^",\s]+)')

# For node lines: nodeID [ ...attrs... ];
NODE_LINE_REGEX = re.compile(r'^(\S+)\s*\[(.*?)\];\s*$')

# For edge lines: leftNode -> rightNode [ ...attrs... ];
EDGE_LINE_REGEX = re.compile(r'^(\S+)\s*->\s*(\S+)\s*\[(.*?)\];\s*$')

# Used to remove port suffixes like ":s", ":n", ":e" from node IDs
PORT_SUFFIX_REGEX = re.compile(r':[a-zA-Z]+$')

# For subgraph label line: label="A_is_valid ()";
LABEL_LINE_REGEX = re.compile(r'^label="([^"]*)"\s*;')

###############################################################################
# parse_attributes: Convert "shape=Mdiamond,style=filled,label="ENTRY""
# into a dict: { "shape": "Mdiamond", "style": "filled", "label": "ENTRY" }
###############################################################################
def parse_attributes(attr_str):
    """
    Given a string of comma-separated attribute pairs (key=value),
    return them as a dict. For example:
       shape=Mdiamond, style=filled, label="ENTRY"
    becomes:
       { "shape": "Mdiamond", "style": "filled", "label": "ENTRY" }
    """
    attrs = {}
    for match in ATTR_PAIR_REGEX.finditer(attr_str):
        key = match.group(1)
        raw_val = match.group(2)  # either "foo" or bar
        if raw_val.startswith('"') and raw_val.endswith('"'):
            # If in quotes, remove the quotes
            val = raw_val[1:-1]
        else:
            val = raw_val
        attrs[key] = val
    return attrs

###############################################################################
# parse_dot_file: Main function to parse a single .dot file
###############################################################################
def parse_dot_file(dot_path):
    with open(dot_path, "r", encoding="utf-8", errors="replace") as f:
        lines = [l.strip() for l in f]

    function_name = None
    nodes = []
    edges = []

    # We'll use the .dot filename (minus extension) as a fallback function name
    fallback_name = os.path.splitext(os.path.basename(dot_path))[0]

    # Attempt to find a subgraph 'label="SomeFunc ()";' line if available
    for line in lines:
        label_match = LABEL_LINE_REGEX.match(line)
        if label_match:
            raw_lbl = label_match.group(1).strip()
            # If it ends with "()", remove those two chars for a cleaner name
            if raw_lbl.endswith("()"):
                raw_lbl = raw_lbl[:-2].strip()
            function_name = raw_lbl
            break

    if not function_name:
        function_name = fallback_name

    # Parse node and edge lines
    for line in lines:
        # Node line check
        node_m = NODE_LINE_REGEX.match(line)
        if node_m:
            node_id_raw = node_m.group(1)   # e.g. fn_0_basic_block_2:s
            attr_part   = node_m.group(2)   # bracket content (shape=Mdiamond,style=..., etc.)

            # Remove port suffix (:s, :n, etc.)
            node_id = PORT_SUFFIX_REGEX.sub('', node_id_raw)

            node_attrs = parse_attributes(attr_part)
            nodes.append({
                "id": node_id,
                "attributes": node_attrs
            })
            continue

        # Edge line check
        edge_m = EDGE_LINE_REGEX.match(line)
        if edge_m:
            from_raw = edge_m.group(1)  # e.g. fn_0_basic_block_0:s
            to_raw   = edge_m.group(2)  # e.g. fn_0_basic_block_2:n
            attr_part= edge_m.group(3)  # bracket content

            from_id = PORT_SUFFIX_REGEX.sub('', from_raw)
            to_id   = PORT_SUFFIX_REGEX.sub('', to_raw)
            edge_attrs = parse_attributes(attr_part)

            edges.append({
                "from": from_id,
                "to": to_id,
                "attributes": edge_attrs
            })
            continue

    # Return the dictionary with function_name, nodes, edges
    return {
        "function_name": function_name,
        "nodes": nodes,
        "edges": edges
    }

###############################################################################
# parse_and_save_dot: parse one .dot file and save as JSON
###############################################################################
def parse_and_save_dot(dot_path, out_json_path):
    parsed = parse_dot_file(dot_path)
    with open(out_json_path, "w", encoding="utf-8") as out_f:
        json.dump(parsed, out_f, indent=2)
    print(f"Parsed '{dot_path}' → '{out_json_path}'")

###############################################################################
# main: parse all .dot files in dot_dir, save results to json_dir
###############################################################################
def main():
    dot_dir = "/home/spire2/SPIRE_GLLeN/Neo4J/dot_files_newDB"
    json_dir = "/home/spire2/SPIRE_GLLeN/Neo4J/json_files"

    # Ensure JSON output directory exists
    os.makedirs(json_dir, exist_ok=True)

    # Process every .dot file in the specified directory
    for filename in os.listdir(dot_dir):
        if filename.endswith(".dot"):
            dot_path = os.path.join(dot_dir, filename)
            base_name = os.path.splitext(filename)[0]
            out_json_path = os.path.join(json_dir, base_name + ".json")
            parse_and_save_dot(dot_path, out_json_path)

if __name__ == "__main__":
    main()
    print("Done parsing all .dot files.")
