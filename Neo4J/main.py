import networkx as nx
import py2neo
from py2neo import Graph


def parse_cfg(cfg_file):
    """Parses a CFG file and extracts nodes and edges.

    Args:
        cfg_file: Path to the CFG file.

    Returns:
        A tuple of nodes and edges.
    """

    nodes = []
    edges = []
    with open(cfg_file, "r") as f:
        for line in f:
            # Adapt parsing logic based on your specific CFG format
            # ...
            if line.startswith("node"):
                node_id, node_label = line.strip().split()
                nodes.append((node_id, {"label": node_label}))
            elif line.startswith("edge"):
                source, target, edge_type = line.strip().split()
                edges.append((source, target, {"type": edge_type}))

    return nodes, edges


def create_neo4j_graph(nodes, edges):
    """Creates a NetworkX graph from nodes and edges.

    Args:
        nodes: A list of nodes with labels.
        edges: A list of edges with source, target, and type.

    Returns:
        A NetworkX DiGraph.
    """

    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph


def import_to_neo4j(graph, uri, auth):
    """Imports a NetworkX graph into a Neo4j database.

    Args:
        graph: The NetworkX graph to import.
        uri: The URI of the Neo4j database.
        auth: Authentication credentials for the Neo4j database.
    """

    neo4j_graph = Graph(uri, auth=auth)

    # Import nodes
    neo4j_graph.run(
        "UNWIND $nodes AS node CREATE (n:Node {id: node.id, label: node.label});",
        nodes=list(graph.nodes(data=True)),
    )

    # Import edges
    neo4j_graph.run(
        "UNWIND $edges AS edge MATCH (n1:Node {id: edge.source}), (n2:Node {id: edge.target}) CREATE (n1)-[:EDGE {type: edge.type}]->(n2);",
        edges=list(graph.edges(data=True)),
    )


def generate_cypher_script(graph):
    """Generates a Cypher script to import the graph into Neo4j.

    Args:
        graph: The NetworkX graph to import.

    Returns:
        A string containing the Cypher script.
    """

    cypher_script = "CREATE "
    for node_id, node_data in graph.nodes(data=True):
        cypher_script += (
            f"(n{node_id}:Node {{id: '{node_id}', label: '{node_data['label']}'}}),"
        )

    cypher_script = cypher_script[:-1] + "\n"

    for source, target, edge_data in graph.edges(data=True):
        cypher_script += (
            f"(n{source})-[:EDGE {{type: '{edge_data['type']}'}}]->(n{target}),\n"
        )

    cypher_script = cypher_script[:-2] + ";"

    return cypher_script


if __name__ == "__main__":
    cfg_file = input("your_cfg_file: ")
    output_file = "import_cfg.cypher"

    nodes, edges = parse_cfg(cfg_file)
    graph = create_neo4j_graph(nodes, edges)

    cypher_script = generate_cypher_script(graph)

    with open(output_file, "w") as f:
        f.write(cypher_script)

    print(f"Cypher script generated: {output_file}")
