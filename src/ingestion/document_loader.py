"""
Document loader for the tender evaluation system
"""

import os
import pdfplumber
import io
from pathlib import Path
from typing import List, Dict, Any
from langchain_core.documents import Document

from src.utils.file_utils import get_files_in_directory, get_file_extension
from src.ocr.ocr_processor import OCRProcessor


class DocumentLoader:
    """Load documents from various sources"""
    
    def __init__(self, ocr_config: Dict[str, Any] = None):
        self.supported_formats = {'.pdf', '.txt'}
        self.ocr_processor = OCRProcessor(ocr_config or {})
    
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
        Load a PDF document using pdfplumber with OCR fallback
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of Document objects
        """
        text = ""
        try:
            # Try to extract text using pdfplumber
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            # If no text was extracted, fallback to OCR
            if not text.strip():
                print(f"No text found in {file_path}, using OCR fallback...")
                text = self.ocr_processor.process_scanned_pdf(file_path)
                
        except Exception as e:
            # If pdfplumber fails, try OCR as fallback
            print(f"pdfplumber failed for {file_path}: {e}. Using OCR fallback...")
            try:
                text = self.ocr_processor.process_scanned_pdf(file_path)
            except Exception as ocr_error:
                raise RuntimeError(f"Failed to extract text from {file_path}: {ocr_error}")
        
        # Create document with extracted text
        return [Document(page_content=text.strip(), metadata={"source": file_path})]
    
    def _load_text_document(self, file_path: str) -> List[Document]:
        """
        Load a text document
        
        Args:
            file_path: Path to the text file
            
        Returns:
            List of Document objects
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return [Document(page_content=text, metadata={"source": file_path})]