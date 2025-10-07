"""
Text processor for parsing and chunking documents
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Dict, Any


class TextProcessor:
    """Process and chunk text documents"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize text processor
        
        Args:
            config: Configuration dictionary with processing settings
        """
        self.config = config or {}
        self.chunk_size = self.config.get('chunk_size', 1000)
        self.chunk_overlap = self.config.get('chunk_overlap', 200)
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks
        
        Args:
            documents: List of Document objects to chunk
            
        Returns:
            List of chunked Document objects
        """
        chunked_documents = []
        
        for doc in documents:
            chunks = self.text_splitter.split_documents([doc])
            chunked_documents.extend(chunks)
        
        return chunked_documents
    
    def extract_metadata(self, document: Document, source_type: str = "organization") -> Dict[str, Any]:
        """
        Extract metadata from a document
        
        Args:
            document: Document to extract metadata from
            source_type: Type of document source ("organization" or "applicant")
            
        Returns:
            Dictionary of metadata
        """
        metadata = document.metadata.copy() if document.metadata else {}
        metadata['source_type'] = source_type
        
        # Add any additional metadata extraction logic here
        # For example, extract document name, creation date, etc.
        
        return metadata