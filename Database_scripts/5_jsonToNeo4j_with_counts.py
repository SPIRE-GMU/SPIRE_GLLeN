#!/usr/bin/env python3
import os
import json
from collections import defaultdict

# Path to your JSON file
JSON_PATH = "/home/spire2/SPIRE_GLLeN/Neo4J/cfg_data.json"


def main():
    if not os.path.exists(JSON_PATH):
        print(f"[ERROR] File not found: {JSON_PATH}")
        return

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        try:
            cfg_data = json.load(f)
        except json.JSONDecodeError:
            print("[ERROR] Could not parse JSON file.")
            return

    # Nested structure: loop_count -> decision_count -> list of node_counts
    dist = defaultdict(lambda: defaultdict(list))

    total_files = 0

    for file_path, func in cfg_data.items():
        node_count = len(func.get("basic_blocks", {}))
        loop_count = len(func.get("loops", []))
        decision_count = len(func.get("decision_nodes", []))

        dist[loop_count][decision_count].append(node_count)
        total_files += 1

    print(f"Analyzed {total_files} function CFGs.\n")
    print(
        "Distribution Summary (Loop Count → Decision Count → Node Count Frequencies):\n"
    )

    for loop_count in sorted(dist.keys()):
        print(f"- {loop_count} loop(s):")
        for decision_count in sorted(dist[loop_count].keys()):
            node_counts = dist[loop_count][decision_count]
            # Frequency count of node values
            node_freq = defaultdict(int)
            for n in node_counts:
                node_freq[n] += 1

            print(f"  ↳ {decision_count} decision(s):")
            for nc in sorted(node_freq.keys()):
                print(f"      • {nc:3d} nodes  =>  {node_freq[nc]} function(s)")


if __name__ == "__main__":
    main()
