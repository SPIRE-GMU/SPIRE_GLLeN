from neo4j import GraphDatabase

# Configuration
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "rootboot"

# Set this to the function you want to delete
TARGET_FUNCTION_UID = "function_2"

# Connect to Neo4j
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def delete_specific_function(tx, uid):
    tx.run("""
        MATCH (f:Function {unique_id: $uid})
        OPTIONAL MATCH (f)-[*]->(sub)
        DETACH DELETE f, sub
    """, uid=uid)

def main():
    print(f"Deleting function '{TARGET_FUNCTION_UID}' from Neo4j...")
    with driver.session() as session:
        session.write_transaction(delete_specific_function, TARGET_FUNCTION_UID)
        print("✅ Deletion complete.")
    driver.close()

if __name__ == "__main__":
    main()
