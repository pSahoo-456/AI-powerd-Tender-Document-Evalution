"""
Streamlit interface for the AI-Powered Tender Evaluation System
"""

import streamlit as st
import os
import tempfile
import pdfplumber
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
from langchain_core.documents import Document


def run_streamlit_app(config_path: str = "./config/config.yaml"):
    """Run the Streamlit application"""
    st.set_page_config(
        page_title="AI-Powered Tender Evaluation System",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("AI-Powered Tender Evaluation System")
    
    # Load configuration
    config_loader = ConfigLoader(config_path)
    config = config_loader.config
    
    # Initialize components (without Ollama model rebuild which causes issues)
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
    
    # File upload section
    st.header("Document Upload")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Organization Documents")
        org_files = st.file_uploader(
            "Upload tender requirements (PDF, TXT)",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            key="org_docs"
        )
    
    with col2:
        st.subheader("Applicant Documents")
        applicant_files = st.file_uploader(
            "Upload applicant proposals (PDF, TXT)",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            key="applicant_docs"
        )
    
    # Evaluation settings
    st.header("Evaluation Settings")
    
    col3, col4 = st.columns(2)
    
    with col3:
        max_applicants = st.number_input(
            "Maximum applicants to evaluate",
            min_value=1,
            max_value=100,
            value=10
        )
    
    with col4:
        min_score = st.number_input(
            "Minimum score threshold",
            min_value=0,
            max_value=100,
            value=70
        )
    
    # Run evaluation button
    if st.button("Run Evaluation", type="primary"):
        if not org_files:
            st.error("Please upload organization documents")
            return
        
        if not applicant_files:
            st.error("Please upload applicant documents")
            return
        
        try:
            with st.spinner("Processing documents and evaluating proposals..."):
                # Process organization documents
                org_documents = process_uploaded_files(org_files, "organization", ocr_processor)
                
                # Process applicant documents
                applicant_documents = process_uploaded_files(applicant_files, "applicant", ocr_processor)
                
                # Generate embeddings
                org_documents = embedding_generator.generate_document_embeddings(org_documents)
                applicant_documents = embedding_generator.generate_document_embeddings(applicant_documents)
                
                # Initialize vector store
                vector_store_manager.initialize_store(org_documents + applicant_documents)
                
                # Apply rule-based filtering
                filtered_applicants = rule_filter.apply_filters(applicant_documents)
                
                # Perform similarity search
                similar_applicants = similarity_searcher.search_applicants_by_requirements(
                    org_documents, filtered_applicants, top_k=max_applicants
                )
                
                # Evaluate with LLM
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
                
                # Generate report
                report_path = report_generator.generate_evaluation_report(
                    org_documents,
                    qualified_applicants,
                    "Tender Evaluation Report"
                )
                
                # Display results
                st.success("Evaluation completed successfully!")
                display_results(qualified_applicants, report_path)
                
        except Exception as e:
            st.error(f"An error occurred during evaluation: {str(e)}")


def process_uploaded_files(uploaded_files, source_type: str, ocr_processor: OCRProcessor) -> List[Document]:
    """Process uploaded files and convert to documents"""
    documents = []
    
    for uploaded_file in uploaded_files:
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_path = tmp_file.name
            
            # Process based on file type
            if uploaded_file.name.endswith('.pdf'):
                # Extract text using pdfplumber with OCR fallback
                text = ""
                try:
                    # Try to extract text using pdfplumber
                    with pdfplumber.open(tmp_path) as pdf:
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"
                    
                    # If no text was extracted, fallback to OCR
                    if not text.strip():
                        st.warning(f"No text found in {uploaded_file.name}, using OCR fallback...")
                        text = ocr_processor.process_scanned_pdf(tmp_path)
                        
                except Exception as e:
                    # If pdfplumber fails, try OCR as fallback
                    st.warning(f"pdfplumber failed for {uploaded_file.name}: {e}. Using OCR fallback...")
                    try:
                        text = ocr_processor.process_scanned_pdf(tmp_path)
                    except Exception as ocr_error:
                        raise RuntimeError(f"Failed to extract text from {uploaded_file.name}: {ocr_error}")
                
                # Create document with extracted text
                doc = Document(page_content=text.strip(), metadata={"source": uploaded_file.name})
                documents.append(doc)
                
            elif uploaded_file.name.endswith('.txt'):
                # Read text file
                with open(tmp_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                doc = Document(page_content=text, metadata={"source": uploaded_file.name})
                documents.append(doc)
            
            # Clean up temporary file
            os.unlink(tmp_path)
            
        except Exception as e:
            st.warning(f"Could not process {uploaded_file.name}: {str(e)}")
    
    return documents


def display_results(evaluated_applicants, report_path: str):
    """Display evaluation results"""
    st.header("Evaluation Results")
    
    if not evaluated_applicants:
        st.info("No applicants met the evaluation criteria.")
        return
    
    # Summary statistics
    st.subheader("Summary")
    avg_score = sum(eval_result.get('score', 0) for _, eval_result in evaluated_applicants) / len(evaluated_applicants)
    st.metric("Average Score", f"{avg_score:.1f}")
    st.metric("Qualified Applicants", len(evaluated_applicants))
    
    # Detailed results
    st.subheader("Detailed Results")
    
    for i, (applicant, eval_result) in enumerate(evaluated_applicants):
        with st.expander(f"Rank {i+1}: {applicant.metadata.get('source', f'Applicant {i+1}')} (Score: {eval_result.get('score', 0)})"):
            st.write(f"**Score:** {eval_result.get('score', 0)}")
            st.write(f"**Explanation:** {eval_result.get('explanation', 'N/A')}")
            st.write(f"**Strengths:** {eval_result.get('strengths', 'N/A')}")
            st.write(f"**Areas for Improvement:** {eval_result.get('improvements', 'N/A')}")
    
    # Report download
    st.subheader("Download Report")
    if os.path.exists(report_path):
        with open(report_path, "rb") as file:
            st.download_button(
                label="Download Evaluation Report",
                data=file,
                file_name=os.path.basename(report_path),
                mime="application/pdf" if report_path.endswith('.pdf') else "application/x-tex"
            )
    else:
        st.warning("Report file not found.")


if __name__ == "__main__":
    run_streamlit_app()