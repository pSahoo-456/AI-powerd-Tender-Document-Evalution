"""
Component tests for the AI-Powered Tender Evaluation System
"""

import os
import sys
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import ConfigLoader
from src.ingestion.document_loader import DocumentLoader
from src.parsing.text_processor import TextProcessor
from src.filtering.rule_filter import RuleFilter


def test_config_loader():
    """Test config loader"""
    print("Testing config loader...")
    config_loader = ConfigLoader("./config/config.yaml")
    assert config_loader.config is not None
    print("✓ Config loader works")


def test_document_loader():
    """Test document loader"""
    print("Testing document loader...")
    
    # Create temporary test documents
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test files
        org_doc = temp_path / "test_doc.txt"
        org_doc.write_text("This is a test document for the tender evaluation system.")
        
        # Test document loading
        doc_loader = DocumentLoader()
        documents = doc_loader.load_documents(str(temp_path))
        assert len(documents) >= 1
        print("✓ Document loader works")


def test_text_processor():
    """Test text processor"""
    print("Testing text processor...")
    
    from langchain_core.documents import Document
    
    # Create test document
    doc = Document(
        page_content="This is a test document with some content that needs to be chunked for processing.",
        metadata={"source": "test"}
    )
    
    # Test config loader
    config_loader = ConfigLoader("./config/config.yaml")
    
    # Test text processing
    text_processor = TextProcessor(config_loader.get('processing', {}))
    chunked_docs = text_processor.chunk_documents([doc])
    assert len(chunked_docs) >= 1
    print("✓ Text processor works")


def test_rule_filter():
    """Test rule filter"""
    print("Testing rule filter...")
    
    from langchain_core.documents import Document
    
    # Create test document
    doc = Document(
        page_content="The applicant proposes a budget of $500,000 and a timeline of 12 months.",
        metadata={"source": "applicant"}
    )
    
    # Test rule filter
    rules_config = {
        'budget': {'max_budget': 1000000},
        'timeline': {'max_duration_months': 24}
    }
    
    rule_filter = RuleFilter(rules_config)
    filtered_docs = rule_filter.apply_filters([doc])
    assert len(filtered_docs) == 1
    print("✓ Rule filter works")


if __name__ == "__main__":
    test_config_loader()
    test_document_loader()
    test_text_processor()
    test_rule_filter()
    print("\nAll component tests passed!")