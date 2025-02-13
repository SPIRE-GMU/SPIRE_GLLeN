#!/usr/bin/env python3
import sys
from neo4j import GraphDatabase

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "rootboot"  # adjust as needed

GDS_GRAPH_NAME = "cfgGraph"
EMBEDDING_PROPERTY = "fastRPEmb"

class GDSFastRPKNN:
    def __init__(self, uri, user, pwd):
        self.driver = GraphDatabase.driver(uri, auth=(user, pwd))

    def close(self):
        self.driver.close()

    def _run_query(self, query, params=None):
        with self.driver.session() as session:
            return list(session.run(query, params or {}))

    def check_if_embeddings_exist(self):
        """
        Check if any :Function node already has EMBEDDING_PROPERTY set (IS NOT NULL).
        If so, skip re-embedding.
        """
        query = f"""
        MATCH (f:Function)
        WHERE f.{EMBEDDING_PROPERTY} IS NOT NULL
        RETURN count(f) AS countWithEmb
        """
        result = self._run_query(query)
        count_with_emb = result[0]["countWithEmb"]
        return count_with_emb > 0

    def try_drop_inmem_graph(self):
        drop_query_1 = f"CALL gds.graph.drop('{GDS_GRAPH_NAME}', false) YIELD graphName"
        drop_query_2 = f"CALL gds.beta.graph.drop('{GDS_GRAPH_NAME}') YIELD graphName"
        try:
            self._run_query(drop_query_1)
            print(f"[INFO] Dropped in-memory graph '{GDS_GRAPH_NAME}' using gds.graph.drop")
        except:
            try:
                self._run_query(drop_query_2)
                print(f"[INFO] Dropped in-memory graph '{GDS_GRAPH_NAME}' using gds.beta.graph.drop")
            except:
                print(f"[WARN] Could not drop in-memory graph '{GDS_GRAPH_NAME}', ignoring.")

    def project_inmem_graph(self):
        """
        Create the in-memory GDS graph from the DB.
        Adjust node labels or relationship types if your schema differs.
        """
        query = f"""
        CALL gds.graph.project(
          '{GDS_GRAPH_NAME}',
          ['Function','BasicBlock','End'],
          {{
            NEXT: {{
              type: 'NEXT',
              orientation: 'UNDIRECTED'
            }}
          }}
        )
        """
        self._run_query(query)
        print(f"[INFO] Created in-memory graph '{GDS_GRAPH_NAME}'")

    def run_fastrp_and_write(self):
        """
        Use iterationWeights + normalizationStrength instead of iterations + normalization.
        iterationWeights: e.g. [0.8, 1.0, 1.0] => 3 iterations
        normalizationStrength: 2 => L2 normalization
        """
        query = f"""
        CALL gds.fastRP.write('{GDS_GRAPH_NAME}', {{
          embeddingDimension: 128,
          iterationWeights: [0.8, 1.0, 1.0],
          normalizationStrength: 2,
          writeProperty: '{EMBEDDING_PROPERTY}'
        }})
        YIELD nodeCount
        """
        result = self._run_query(query)
        if result:
            row = result[0]
            print("[INFO] FastRP done:")
            print(f" - nodeCount: {row['nodeCount']}")

    def run_knn_write(self):
        """
        Once we have embeddings, run kNN in 'write' mode to create :SIMILAR relationships.
        We'll only YIELD nodesCompared, relationshipsWritten.
        """
        query = f"""
        CALL gds.knn.write('{GDS_GRAPH_NAME}', {{
          topK: 3,
          nodeProperties: ['{EMBEDDING_PROPERTY}'],
          similarityCutoff: 0.0,
          writeRelationshipType: 'SIMILAR',
          writeProperty: 'score'
        }})
        YIELD nodesCompared, relationshipsWritten
        """
        result = self._run_query(query)
        if result:
            row = result[0]
            print("[INFO] kNN done:")
            print(f" - nodesCompared: {row['nodesCompared']}")
            print(f" - relationshipsWritten: {row['relationshipsWritten']}")

    def demo_top_similar(self, limit=10):
        """
        Show top :SIMILAR relationships from the DB by descending score
        """
        query = f"""
        MATCH (n)-[rel:SIMILAR]->(m)
        RETURN n.id AS nId, m.id AS mId, rel.score AS score
        ORDER BY score DESC
        LIMIT {limit}
        """
        rows = self._run_query(query)
        print("[INFO] Top SIMILAR relationships:")
        for r in rows:
            print(f"   {r['nId']} -> {r['mId']} [score={r['score']:.4f}]")

def main():
    if len(sys.argv) < 2:
        print("Usage: python 05_knn_search.py <cfgFileArg>")
        sys.exit(1)

    dummy_arg = sys.argv[1]

    gds_tool = GDSFastRPKNN(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        # 1) Check if we already have embeddings
        has_emb = gds_tool.check_if_embeddings_exist()
        if has_emb:
            print("[INFO] Embeddings exist, skipping FastRP creation.")
        else:
            print("[INFO] No existing embeddings. Creating them via FastRP...")
            gds_tool.try_drop_inmem_graph()
            gds_tool.project_inmem_graph()
            gds_tool.run_fastrp_and_write()

        # 2) We'll run kNN no matter what
        gds_tool.try_drop_inmem_graph()
        gds_tool.project_inmem_graph()
        gds_tool.run_knn_write()
        gds_tool.demo_top_similar(10)

    finally:
        gds_tool.close()

if __name__ == "__main__":
    main()
