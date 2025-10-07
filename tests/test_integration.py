"""
Integration tests for the AI-Powered Tender Evaluation System
"""

import os
import sys
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils.config_loader import ConfigLoader
from src.ingestion.document_loader import DocumentLoader
from src.parsing.text_processor import TextProcessor
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.vector_db.vector_store import VectorStoreManager
from src.search.similarity_search import SimilaritySearcher


def test_system_components():
    """Test that all system components can be initialized"""
    print("Testing system component initialization...")
    
    # Test config loader
    config_loader = ConfigLoader("./config/config.yaml")
    assert config_loader.config is not None
    print("✓ Config loader initialized")
    
    # Test document loader
    doc_loader = DocumentLoader()
    assert doc_loader is not None
    print("✓ Document loader initialized")
    
    # Test text processor
    text_processor = TextProcessor(config_loader.get('processing', {}))
    assert text_processor is not None
    print("✓ Text processor initialized")
    
    # Test embedding generator
    embedding_generator = EmbeddingGenerator(config_loader.get('ollama', {}))
    assert embedding_generator is not None
    print("✓ Embedding generator initialized")
    
    # Test vector store manager
    vector_store_manager = VectorStoreManager(config_loader.get('vector_db', {}))
    assert vector_store_manager is not None
    print("✓ Vector store manager initialized")
    
    # Test similarity searcher
    similarity_searcher = SimilaritySearcher(vector_store_manager)
    assert similarity_searcher is not None
    print("✓ Similarity searcher initialized")
    
    print("All components initialized successfully!")


def test_document_processing():
    """Test document processing workflow"""
    print("\nTesting document processing workflow...")
    
    # Create temporary test documents
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test files
        org_doc = temp_path / "org_requirements.txt"
        org_doc.write_text("The organization requires experience in software development and cloud platforms.")
        
        applicant_doc = temp_path / "applicant_proposal.txt"
        applicant_doc.write_text("We have 10 years of experience in software development and cloud platforms like AWS and Azure.")
        
        # Test document loading
        doc_loader = DocumentLoader()
        org_documents = doc_loader.load_documents(str(temp_path))
        assert len(org_documents) >= 1
        print("✓ Documents loaded successfully")
        
        # Test text processing
        config_loader = ConfigLoader("./config/config.yaml")
        text_processor = TextProcessor(config_loader.get('processing', {}))
        chunked_docs = text_processor.chunk_documents(org_documents)
        assert len(chunked_docs) >= 1
        print("✓ Documents chunked successfully")


if __name__ == "__main__":
    test_system_components()
    test_document_processing()
    print("\nAll integration tests passed!")