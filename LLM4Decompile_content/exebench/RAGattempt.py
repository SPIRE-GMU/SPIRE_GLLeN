import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict
import pickle
from pathlib import Path

class FAISSDocStore:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the FAISS document store with a specified sentence transformer model.
        
        Args:
            model_name (str): Name of the sentence transformer model to use
        """
        self.encoder = SentenceTransformer(model_name)
        self.index = None
        self.snippets = []
        self.dimension = None
        
    def add_texts(self, snippets: List[str]) -> None:
        """
        Add text snippets to the document store.
        
        Args:
            snippets (Lis[str]): List of text snippets to add
        """
        # Store original snippets
        self.snippets.extend(snippets)
        
        # Convert texts to embeddings
        embeddings = self.encoder.encode(snippets, convert_to_tensor=True)
        embeddings_np = embeddings.cpu().numpy()
        
        # Initialize FAISS index if not already done
        if self.index is None:
            self.dimension = embeddings_np.shape[1]
            self.index = faiss.IndexFlatL2(self.dimension)
        
        # Add vectors to the index
        self.index.add(embeddings_np.astype(np.float32))
        
    def search(self, query: str, k: int = 5) -> List[Dict[str, any]]:
        """
        Search for similar documents given a query string.
        
        Args:
            query (str): Query text to search for
            k (int): Number of results to return
            
        Returns:
            List[Dict]: List of dictionaries containing search results with scores
        """
        # Convert query to embedding
        query_embedding = self.encoder.encode([query], convert_to_tensor=True)
        query_embedding_np = query_embedding.cpu().numpy().astype(np.float32)
        
        # Perform search
        distances, indices = self.index.search(query_embedding_np, k)
        
        # Format results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.snippets):  # Ensure valid index
                results.append({
                    'snippet': self.snippets[idx],
                    'score': float(dist),
                    'index': int(idx)
                })
        
        return results
    
    def save(self, directory: str) -> None:
        """
        Save the document store to disk.
        
        Args:
            directory (str): Directory to save the index and metadata
        """
        save_dir = Path(directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(save_dir / "index.faiss"))
        
        # Save snippets and other metadata
        metadata = {
            'snippets': self.snippets,
            'dimension': self.dimension
        }
        with open(save_dir / "metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
            
    @classmethod
    def load(cls, directory: str) -> 'FAISSDocStore':
        """
        Load a document store from disk.
        
        Args:
            directory (str): Directory containing the saved index and metadata
            
        Returns:
            FAISSDocStore: Loaded document store instance
        """
        load_dir = Path(directory)
        
        # Create instance
        instance = cls()
        
        # Load FAISS index
        instance.index = faiss.read_index(str(load_dir / "index.faiss"))
        
        # Load metadata
        with open(load_dir / "metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
            instance.snippets = metadata['snippets']
            instance.dimension = metadata['dimension']
            
        return instance

# Example usage
if __name__ == "__main__":
    # Sample text snippets
    snippets = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "Python is a versatile programming language",
        "Natural language processing involves understanding text",
    ]
    
    # Create and populate document store
    doc_store = FAISSDocStore()
    doc_store.add_texts(snippets)
    
    # Perform a search
    results = doc_store.search("artificial intelligence", k=2)
    for result in results:
        print(f"Score: {result['score']:.4f} | Text: {result['snippet']}")
    
    # Save and load example
    doc_store.save("./docstore")
    loaded_store = FAISSDocStore.load("./docstore")t
