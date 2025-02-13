# Author: Craig Kimball

# Data Imports
import os
from dotenv import load_dotenv
import numpy as np
import subprocess

# Model Imports
import faiss
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM


class CodeEmbeddingSearch:
    def __init__(self, index_dim=768):
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = AutoModel.from_pretrained("microsoft/codebert-base")

        # FAISS index for embedding storage
        self.index = faiss.IndexFlatL2(index_dim)

    def save_index(self, file_path="assemblyIndex.index"):
        """Save FAISS index to a file."""
        faiss.write_index(self.index, file_path)

    @staticmethod
    def load_index(file_path="assemblyIndex.index"):
        """Load FAISS index from a file."""
        return faiss.read_index(file_path)

    def embed_code(self, code_snippet):
        """Create an embedding for a code snippet."""
        inputs = self.tokenizer(
            code_snippet, return_tensors="pt", truncation=True, max_length=128
        )
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy().squeeze()

    def generate_embeddings(self, assembly_files, asm_dir="asm_C/"):
        """Generate embeddings for assembly code files and add them to the FAISS index."""
        assembly_code_list = [
            open(os.path.join(asm_dir, file), "r").read() for file in assembly_files
        ]
        # Embed all assembly code snippets
        assembly_embeddings = np.array(
            [self.embed_code(code) for code in assembly_code_list]
        )
        self.index.add(assembly_embeddings)

    def make_search(self, input_code, c_code_files, c_code_dir="asm_C/"):
        """Search for the closest C code match for a given assembly code."""
        c_code_outputs = [
            open(os.path.join(c_code_dir, file), "r").read() for file in c_code_files
        ]
        query_embedding = self.embed_code(input_code).reshape(1, -1)
        k = 1  # Number of nearest neighbors to retrieve
        distances, indices = self.index.search(query_embedding, k)

        return [c_code_outputs[i] for i in indices[0]]


# Neo4j class placeholder for future implementation
class Neo4jHandler:
    def __init__(self, uri, auth):
        from neo4j import GraphDatabase  # Import here to decouple from the core logic

        self.driver = GraphDatabase.driver(uri, auth=auth)

    def verify_connection(self):
        """Verify the Neo4j database connection."""
        with self.driver.session() as session:
            session.run("RETURN 1")
        print("Database connection verified.")

    # Add Neo4j-related methods here as needed
    def PURGE(self):
        query = """
        MATCH (n) DETACH DELETE n
        """

        input("Confirm Action")
        records, summary, keys = self.driver.execute_query(query, database_="neo4j")

    def generate_cfg_c(self, c_file_path):
        gcc_cfg_command = [
            "gcc",  # Adjusted to use gcc 11.4.0 for ubuntu
            "-fdump-tree-all-graph",
            "-c",  # Compile only, do not link (suitable for files without main)
            c_file_path,
            "-o",
            os.path.join(temp_dir, f"{filename}.out"),
        ]

        try:
            # Run the GCC command to create the CFG
            result_cfg = subprocess.run(
                gcc_cfg_command, check=True, capture_output=True, text=True
            )
        except Exception as e:
            print("Oops")


# Usage Example
if __name__ == "__main__":
    # Define file lists
    asm_files = [
        "factorial.s",
        "fibonacci.s",
        "floating_mult.s",
        "hello_world.s",
        "natural_sum_recursion.s",
        "num_swap.s",
        "powers_loop.s",
        "quadratic_roots.s",
        "quotient_remainder.s",
        "recursive_factorial.s",
    ]

    c_files = [
        "factorial.c",
        "fibonacci.c",
        "floating_mult.c",
        "hello_world.c",
        "natural_sum_recursion.c",
        "num_swap.c",
        "powers_loop.c",
        "quadratic_roots.c",
        "quotient_remainder.c",
        "recursive_factorial.c",
    ]

    file_directory = "/home/spire2/SPIRE_GLLeN/ragPipeline/asm_C"
    asm_files = []

    # Check if the provided directory path exists
    if os.path.isdir(file_directory):
        # Iterate over all files in the directory
        for filename in os.listdir(file_directory):
            if filename.endswith(".s"):  # Check for .asm files
                asm_files.append(filename)  # Store the file name in the list

    # for name in asm_files:
    #     print(name)

    c_files = [x.replace(".s", ".c") for x in asm_files]

    # for name in c_files:
    #     print(name)

    """ Initialize the CodeEmbeddingSearch search tool for vector based operations """
    search_tool = CodeEmbeddingSearch()

    """ Initialize neo4j variables """
    load_dotenv()
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    NEO4J_URI = os.getenv("NEO4J_CONNECTION_URI")

    """ Initialize neo4j database handler """
    # neo_tool = Neo4jHandler(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    # neo_tool.verify_connection()

    """ Generate and save embeddings """
    search_tool.generate_embeddings(asm_files)
    # search_tool.save_index()
    # search_tool.index = search_tool.load_index()

    """ Filepath to file you wish to decompile """
    asm_file_path = "asm_C/hello_world.s"

    with open(asm_file_path, "r") as asm_file:
        input_asm_code = asm_file.read()

    """ perform vector search """
    matched_c_code = search_tool.make_search(input_asm_code, c_files)

    print("\n\nReturned C:\n\n")
    print("\n".join(matched_c_code))

    """ Prepare input for a DeepSeek Coder or other downstream tasks """
    # Only DeepSeek Handled Here
    # deepseek_model_path = "deepseek-ai/deepseek-coder-6.7b-base"
    # deepseek_model_path = "deepseek-ai/deepseek-coder-1.3b-base"
    deepseek_model_path = "deepseek-ai/deepseek-coder-6.7b-instruct"
    # deepseek_model_path = "deepseek-ai/deepseek-coder-1.3b-instruct"
    # deepseek_model_path = "deepseek-ai/deepseek-coder-6.7b-base"

    tokenizer_chat = AutoTokenizer.from_pretrained(
        deepseek_model_path, trust_remote_code=True
    )
    model_chat = AutoModelForCausalLM.from_pretrained(
        deepseek_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).cuda()

    # assemble files to format model can understand

    input_text = (
        "# Assembly Code:\n"
        + input_asm_code
        + "\n\n# C Code:\n"
        + "\n".join(matched_c_code)
        + "\n\n# Decompile the above Assembly code given the C code as context"
    )

    # input_text = "#write a single bubble sort algorith in python"

    # Tokenize and generate output
    inputs = tokenizer_chat(input_text, return_tensors="pt").to(model_chat.device)
    outputs = model_chat.generate(**inputs, max_length=2048)

    print("\n\n\nResponse:\n\n\n")
    print(tokenizer_chat.decode(outputs[0], skip_special_tokens=True))
