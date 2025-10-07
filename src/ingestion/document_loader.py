"""
Document loader for the tender evaluation system
"""

import os
from pathlib import Path
from typing import List, Dict, Any
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.documents import Document

from src.utils.file_utils import get_files_in_directory, get_file_extension


class DocumentLoader:
    """Load documents from various sources"""
    
    def __init__(self):
        self.supported_formats = {'.pdf', '.txt'}
    
    def load_documents(self, directory_path: str) -> List[Document]:
        """
        Load all supported documents from a directory
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            List of Document objects
        """
        documents = []
        files = get_files_in_directory(directory_path)
        
        for file_path in files:
            extension = get_file_extension(file_path)
            if extension in self.supported_formats:
                try:
                    docs = self._load_document(str(file_path))
                    documents.extend(docs)
                except Exception as e:
                    print(f"Warning: Could not load {file_path}: {e}")
        
        return documents
    
    def _load_document(self, file_path: str) -> List[Document]:
        """
        Load a single document based on its extension
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of Document objects
        """
        extension = get_file_extension(file_path)
        
        if extension == '.pdf':
            return self._load_pdf_document(file_path)
        elif extension == '.txt':
            return self._load_text_document(file_path)
        else:
            raise ValueError(f"Unsupported document format: {extension}")
    
    def _load_pdf_document(self, file_path: str) -> List[Document]:
        """
        Load a PDF document
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of Document objects
        """
        loader = PyMuPDFLoader(file_path)
        return loader.load()
    
    def _load_text_document(self, file_path: str) -> List[Document]:
        """
        Load a text document
        
        Args:
            file_path: Path to the text file
            
        Returns:
            List of Document objects
        """
        loader = TextLoader(file_path, encoding='utf-8')
        return loader.load()