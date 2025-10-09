"""
CLI interface for the AI-Powered Tender Evaluation System
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List

from src.utils.config_loader import ConfigLoader
from src.ingestion.document_loader import DocumentLoader
from src.ocr.ocr_processor import OCRProcessor
from src.parsing.text_processor import TextProcessor
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.vector_db.vector_store import VectorStoreManager
from src.search.similarity_search import SimilaritySearcher
from src.filtering.rule_filter import RuleFilter
from src.evaluation.llm_evaluator import LLMEvaluator
from src.reporting.report_generator import ReportGenerator


def run_cli_app(config_path: str = "./config/config.yaml"):
    """Run the CLI application"""
    print("AI-Powered Tender Evaluation System")
    print("=" * 50)
    
    # Load configuration
    config_loader = ConfigLoader(config_path)
    config = config_loader.config
    
    # Get paths from config or use defaults
    org_docs_path = config_loader.get('paths.org_documents', './data/org_documents')
    applicant_docs_path = config_loader.get('paths.applicant_documents', './data/applicant_documents')
    
    print(f"Organization documents path: {org_docs_path}")
    print(f"Applicant documents path: {applicant_docs_path}")
    
    # Check if directories exist and have files
    if not os.path.exists(org_docs_path) or not os.listdir(org_docs_path):
        print(f"Warning: No organization documents found in {org_docs_path}")
        return
    
    if not os.path.exists(applicant_docs_path) or not os.listdir(applicant_docs_path):
        print(f"Warning: No applicant documents found in {applicant_docs_path}")
        return
    
    # Initialize components
    print("Initializing system components...")
    ocr_config = config.get('ocr', {})
    doc_loader = DocumentLoader(ocr_config)
    ocr_processor = OCRProcessor(ocr_config)
    text_processor = TextProcessor(config.get('processing', {}))
    embedding_generator = EmbeddingGenerator(config.get('ollama', {}))
    vector_store_manager = VectorStoreManager(config.get('vector_db', {}))
    similarity_searcher = SimilaritySearcher(vector_store_manager)
    rule_filter = RuleFilter(config.get('rules', {}))
    llm_evaluator = LLMEvaluator(config.get('ollama', {}))
    report_generator = ReportGenerator()
    
    try:
        # Load documents
        print("Loading organization documents...")
        org_documents = doc_loader.load_documents(org_docs_path)
        print(f"Loaded {len(org_documents)} organization documents")
        
        print("Loading applicant documents...")
        applicant_documents = doc_loader.load_documents(applicant_docs_path)
        print(f"Loaded {len(applicant_documents)} applicant documents")
        
        # Process documents with OCR if needed
        print("Processing documents with OCR if needed...")
        # In a real implementation, you would check if PDFs are scanned and process accordingly
        
        # Generate embeddings
        print("Generating embeddings...")
        org_documents = embedding_generator.generate_document_embeddings(org_documents)
        applicant_documents = embedding_generator.generate_document_embeddings(applicant_documents)
        
        # Initialize vector store
        print("Initializing vector store...")
        vector_store_manager.initialize_store(org_documents + applicant_documents)
        
        # Apply rule-based filtering
        print("Applying rule-based filtering...")
        filtered_applicants = rule_filter.apply_filters(applicant_documents)
        print(f"Filtered applicants: {len(filtered_applicants)} remaining")
        
        # Get processing parameters
        max_applicants = config_loader.get('evaluation.max_applicants', 10)
        min_score = config_loader.get('evaluation.min_score', 70)
        
        # Perform similarity search
        print("Performing similarity search...")
        similar_applicants = similarity_searcher.search_applicants_by_requirements(
            org_documents, filtered_applicants, top_k=max_applicants
        )
        print(f"Found {len(similar_applicants)} similar applicants")
        
        # Evaluate with LLM
        print("Evaluating with LLM...")
        evaluated_applicants = llm_evaluator.evaluate_applicants(
            org_documents, 
            [doc for doc, _ in similar_applicants],
            max_applicants=max_applicants
        )
        
        # Filter by minimum score
        qualified_applicants = [
            (doc, eval_result) for doc, eval_result in evaluated_applicants
            if eval_result.get('score', 0) >= min_score
        ]
        
        print(f"Qualified applicants: {len(qualified_applicants)}")
        
        # Display results
        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        
        if not qualified_applicants:
            print("No applicants met the evaluation criteria.")
            return
        
        # Sort by score
        qualified_applicants.sort(key=lambda x: x[1].get('score', 0), reverse=True)
        
        # Display ranked results
        for i, (applicant, eval_result) in enumerate(qualified_applicants):
            print(f"\nRank {i+1}:")
            print(f"  Score: {eval_result.get('score', 0)}")
            print(f"  Explanation: {eval_result.get('explanation', 'N/A')}")
            print(f"  Strengths: {eval_result.get('strengths', 'N/A')}")
            print(f"  Areas for Improvement: {eval_result.get('improvements', 'N/A')}")
        
        # Generate report
        print("\nGenerating evaluation report...")
        report_path = report_generator.generate_evaluation_report(
            org_documents,
            qualified_applicants,
            "Tender Evaluation Report"
        )
        
        print(f"Report generated: {report_path}")
        print("\nEvaluation completed successfully!")
        
    except Exception as e:
        print(f"An error occurred during evaluation: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    run_cli_app()