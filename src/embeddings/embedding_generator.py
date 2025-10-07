"""
Embedding generator using Ollama
"""

from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from typing import List, Dict, Any


class EmbeddingGenerator:
    """Generate embeddings for documents using Ollama"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize embedding generator
        
        Args:
            config: Configuration dictionary with Ollama settings
        """
        self.config = config or {}
        self.base_url = self.config.get('base_url', 'http://localhost:11434')
        self.model = self.config.get('embedding_model', 'nomic-embed-text-v2')
        
        # Try to initialize Ollama embeddings with error handling
        self.embeddings = None
        try:
            self.embeddings = OllamaEmbeddings(
                model=self.model,
                base_url=self.base_url
            )
        except Exception as e:
            print(f"Warning: Could not initialize Ollama embeddings: {e}")
            print("Embedding generation will be simulated.")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        # If embeddings are not available, simulate them
        if not self.embeddings:
            return self._simulate_embeddings(texts)
        
        try:
            return self.embeddings.embed_documents(texts)
        except Exception as e:
            if "not found" in str(e) or "404" in str(e):
                print(f"Error: Model '{self.model}' not found. Please pull it first with:")
                print(f"  ollama pull {self.model}")
                print("Falling back to simulated embeddings...")
                return self._simulate_embeddings(texts)
            else:
                print(f"Warning: Embedding generation failed: {e}")
                print("Falling back to simulated embeddings...")
                return self._simulate_embeddings(texts)
    
    def generate_document_embeddings(self, documents: List[Document]) -> List[Document]:
        """
        Generate embeddings for documents and add them to metadata
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of Document objects with embeddings in metadata
        """
        # Extract texts from documents
        texts = [doc.page_content for doc in documents]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Add embeddings to document metadata
        embedded_documents = []
        for i, doc in enumerate(documents):
            # Create a copy of the document with embeddings in metadata
            embedded_doc = Document(
                page_content=doc.page_content,
                metadata=doc.metadata.copy() if doc.metadata else {}
            )
            embedded_doc.metadata['embedding'] = embeddings[i]
            embedded_documents.append(embedded_doc)
        
        return embedded_documents
    
    def _simulate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Simulate embeddings when Ollama is not available
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of simulated embedding vectors
        """
        print("Simulating embeddings (Ollama not available)")
        import random
        
        # Generate random embeddings (384 dimensions for nomic-embed-text)
        simulated_embeddings = []
        for text in texts:
            # Generate a random 384-dimensional vector
            embedding = [random.uniform(-1, 1) for _ in range(384)]
            simulated_embeddings.append(embedding)
        
        return simulated_embeddings