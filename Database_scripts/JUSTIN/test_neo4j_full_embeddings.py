#!/usr/bin/env python3
import json
import math
from neo4j import GraphDatabase

# Neo4j connection details (update if needed)
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "rootboot"

STATS_FILE = "extracted_structure_stats.json"

def compute_normalized(vector):
    """Compute the L2-normalized version of a vector."""
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        return vector
    return [x / norm for x in vector]

def main():
    print("Connecting to Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    # Query to extract the needed fields from Function nodes.
    # Assumes that node_count, edge_count, loop_count and decision_count have been stored.
    query = """
    MATCH (f:Function)
    WHERE f.node_count IS NOT NULL AND f.edge_count IS NOT NULL 
      AND f.loop_count IS NOT NULL AND f.decision_count IS NOT NULL
    RETURN f.unique_id AS unique_id, f.node_count AS node_count, 
           f.edge_count AS edge_count, f.loop_count AS loop_count, 
           f.decision_count AS decision_count
    """
    
    functions_data = []
    with driver.session() as session:
        result = session.run(query)
        for record in result:
            functions_data.append({
                "unique_id": record["unique_id"],
                "node_count": record["node_count"],
                "edge_count": record["edge_count"],
                "loop_count": record["loop_count"],
                "decision_count": record["decision_count"]
            })
    print(f"Retrieved {len(functions_data)} functions from Neo4j.")

    # Build the four different embeddings for each function.
    for func in functions_data:
        # Convert numeric fields to floats for safety
        node_count = float(func["node_count"])
        edge_count = float(func["edge_count"])
        loop_count = float(func["loop_count"])
        decision_count = float(func["decision_count"])
        
        embed_basic = [node_count, edge_count]
        embed_with_loops = [node_count, edge_count, loop_count]
        embed_full = [node_count, edge_count, loop_count, decision_count]
        embed_norm = compute_normalized(embed_full)
        
        func["embed_basic"] = embed_basic
        func["embed_with_loops"] = embed_with_loops
        func["embed_full"] = embed_full
        func["embed_norm"] = embed_norm
        
        print(f"Function {func['unique_id']}:")
        print(f"  Basic:         {embed_basic}")
        print(f"  With loops:    {embed_with_loops}")
        print(f"  Full:          {embed_full}")
        print(f"  Normalized:    {embed_norm}")

    # Save the extracted data to a stats file for reference.
    with open(STATS_FILE, "w") as stats_out:
        json.dump(functions_data, stats_out, indent=2)
    print(f"Structural stats saved to {STATS_FILE}.")

    # Update Neo4j with the computed embeddings
    update_query = """
    MATCH (f:Function {unique_id: $unique_id})
    SET f.embed_basic = $embed_basic,
        f.embed_with_loops = $embed_with_loops,
        f.embed_full = $embed_full,
        f.embed_norm = $embed_norm
    """
    with driver.session() as session:
        for idx, func in enumerate(functions_data, start=1):
            session.run(
                update_query,
                unique_id=func["unique_id"],
                embed_basic=json.dumps(func["embed_basic"]),
                embed_with_loops=json.dumps(func["embed_with_loops"]),
                embed_full=json.dumps(func["embed_full"]),
                embed_norm=json.dumps(func["embed_norm"])
            )
            if idx % 50 == 0:
                print(f"Updated embeddings for {idx} functions...")
                
    driver.close()
    print("All functions updated with new embeddings.")

if __name__ == "__main__":
    main()
