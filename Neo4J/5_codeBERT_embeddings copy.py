#!/usr/bin/env python3
import os
import json
import torch
import numpy as np
from neo4j import GraphDatabase
from transformers import RobertaTokenizer, RobertaModel
from sklearn.metrics.pairwise import cosine_similarity

# Model configuration: use "microsoft/graphcodebert-base" for graph-based comparison.
# You can switch to "microsoft/codebert-base" if needed.
MODEL_NAME = "microsoft/graphcodebert-base"

print("[INFO] Loading model:", MODEL_NAME)
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
model = RobertaModel.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print("[INFO] Model loaded.")

def get_embedding(code_str, max_len=512):
    """
    Get a normalized embedding for the input code.
    Uses mean pooling over token embeddings.
    """
    inputs = tokenizer(code_str, max_length=max_len, truncation=True, return_tensors="pt")
    for key, value in inputs.items():
        inputs[key] = value.to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        token_embeddings = outputs.last_hidden_state[0]  # shape: (seq_len, hidden_size)
        pooled_embedding = token_embeddings.mean(dim=0)
    emb = pooled_embedding.cpu().numpy()
    norm = np.linalg.norm(emb)
    return emb / norm if norm != 0 else emb

def compute_similarity(query_embedding, db_embeddings):
    """
    Compute cosine similarity between a query embedding and a list of embeddings.
    """
    db_embeddings = np.array(db_embeddings)
    query_embedding = np.array(query_embedding).reshape(1, -1)
    sims = cosine_similarity(query_embedding, db_embeddings)
    return sims[0]

# Neo4j connection details
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "rootboot"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def store_embedding_in_neo4j(function_name, embedding):
    """
    Store the normalized embedding in a Neo4j node.
    The node must have a property function_name that matches the given name.
    """
    with driver.session() as session:
        session.run(
            """
            MATCH (f:Function {function_name: $fname})
            SET f.embedding = $emb
            RETURN f.function_name AS name
            """,
            fname=function_name,
            emb=embedding.tolist()
        )

if __name__ == "__main__":
    # Example usage: compute embedding for a code file and compare with sample DB embeddings.
    code_file = "example_code.c"
    if not os.path.isfile(code_file):
        print(f"[ERROR] Code file '{code_file}' not found.")
    else:
        try:
            with open(code_file, "r", encoding="utf-8") as f:
                code_str = f.read()
            query_emb = get_embedding(code_str)
            print("[INFO] Embedding computed.")
            
            # For demonstration, assume a list of embeddings from the database.
            # Replace with actual embeddings loaded from Neo4j.
            db_embeddings = [query_emb]
            
            sims = compute_similarity(query_emb, db_embeddings)
            print("[INFO] Similarity scores:", sims)
            
            # Optionally store the embedding in Neo4j.
            store_embedding_in_neo4j("example_function", query_emb)
            print("[INFO] Embedding stored in Neo4j.")
        except Exception as e:
            print("[ERROR]", str(e))
