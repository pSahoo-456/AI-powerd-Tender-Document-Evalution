"""
Vector store for managing document embeddings
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import faiss
import numpy as np
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import OllamaEmbeddings


class VectorStoreManager:
    """Manage vector storage for document embeddings"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize vector store manager
        
        Args:
            config: Configuration dictionary with vector DB settings
        """
        self.config = config or {}
        self.db_type = self.config.get('type', 'faiss')
        self.persist_directory = self.config.get('persist_directory', './data/vectorstore')
        
        # Ensure persist directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize embeddings
        ollama_config = self.config.get('ollama', {})
        self.embeddings = OllamaEmbeddings(
            model=ollama_config.get('embedding_model', 'nomic-embed-text'),
            base_url=ollama_config.get('base_url', 'http://localhost:11434')
        )
        
        self.vector_store = None
    
    def initialize_store(self, documents: List[Document] = None):
        """
        Initialize the vector store with documents
        
        Args:
            documents: List of Document objects to initialize with (optional)
        """
        if self.db_type == 'faiss':
            self._initialize_faiss(documents)
        elif self.db_type == 'chroma':
            self._initialize_chroma(documents)
        else:
            raise ValueError(f"Unsupported vector database type: {self.db_type}")
    
    def _initialize_faiss(self, documents: List[Document] = None):
        """
        Initialize FAISS vector store
        
        Args:
            documents: List of Document objects to initialize with (optional)
        """
        persist_path = Path(self.persist_directory) / "faiss_index"
        
        if documents:
            # Create new FAISS store with documents
            self.vector_store = FAISS.from_documents(
                documents, 
                self.embeddings
            )
            
            # Save the store
            self.vector_store.save_local(str(persist_path))
        elif persist_path.exists():
            # Load existing FAISS store
            self.vector_store = FAISS.load_local(
                str(persist_path), 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            # Create empty FAISS store
            from langchain_core.documents import Document
            dummy_doc = Document(page_content="Dummy document for initialization", metadata={})
            self.vector_store = FAISS.from_documents([dummy_doc], self.embeddings)
            self.vector_store.save_local(str(persist_path))
    
    def _initialize_chroma(self, documents: List[Document] = None):
        """
        Initialize Chroma vector store
        
        Args:
            documents: List of Document objects to initialize with (optional)
        """
        persist_path = Path(self.persist_directory) / "chroma_db"
        
        if documents:
            # Create new Chroma store with documents
            self.vector_store = Chroma.from_documents(
                documents,
                self.embeddings,
                persist_directory=str(persist_path)
            )
        else:
            # Create or load existing Chroma store
            self.vector_store = Chroma(
                persist_directory=str(persist_path),
                embedding_function=self.embeddings
            )
    
    def add_documents(self, documents: List[Document]):
        """
        Add documents to the vector store
        
        Args:
            documents: List of Document objects to add
        """
        if not self.vector_store:
            self.initialize_store(documents)
        else:
            self.vector_store.add_documents(documents)
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Perform similarity search on the vector store
        
        Args:
            query: Query string to search for
            k: Number of results to return
            
        Returns:
            List of Document objects
        """
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized")
        
        return self.vector_store.similarity_search(query, k=k)
    
    def save_store(self):
        """Save the vector store to disk"""
        if not self.vector_store:
            return
            
        if self.db_type == 'faiss':
            persist_path = Path(self.persist_directory) / "faiss_index"
            self.vector_store.save_local(str(persist_path))
        # Chroma persists automatically