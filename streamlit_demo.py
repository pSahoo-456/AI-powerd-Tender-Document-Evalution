"""
Streamlit demo for the AI-Powered Tender Evaluation System
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config_loader import ConfigLoader
from src.ingestion.document_loader import DocumentLoader
from src.parsing.text_processor import TextProcessor
from src.filtering.rule_filter import RuleFilter


def run_demo_app():
    """Run the Streamlit demo application"""
    st.set_page_config(
        page_title="AI-Powered Tender Evaluation System - Demo",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("AI-Powered Tender Evaluation System")
    st.subheader("Demonstration Interface")
    
    # Sidebar with system info
    st.sidebar.header("System Information")
    st.sidebar.info("""
    This is a demonstration of the AI-Powered Tender Evaluation System.
    
    In a full implementation, this system would:
    - Process tender documents with OCR if needed
    - Generate embeddings using Ollama
    - Perform semantic similarity search
    - Apply rule-based filtering
    - Evaluate proposals with LLM
    - Generate professional PDF reports
    """)
    
    # Main content
    st.header("System Workflow Demonstration")
    
    # Step 1: Document Upload
    st.subheader("1. Document Upload")
    st.info("In a full implementation, you would upload organization requirements and applicant proposals here.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Organization Documents**")
        st.markdown("- Sample Tender Document (Digital Transformation Initiative)")
        
    with col2:
        st.markdown("**Applicant Documents**")
        st.markdown("- Sample Applicant Proposal (TechSolutions Inc.)")
    
    # Step 2: Processing
    st.subheader("2. Document Processing")
    
    if st.button("Process Documents"):
        with st.spinner("Processing documents..."):
            # Load configuration
            config_loader = ConfigLoader("./config/config.yaml")
            config = config_loader.config
            
            # Load documents
            doc_loader = DocumentLoader()
            org_documents = doc_loader.load_documents("./data/org_documents")
            applicant_documents = doc_loader.load_documents("./data/applicant_documents")
            
            # Process text
            text_processor = TextProcessor(config.get('processing', {}))
            chunked_org_docs = text_processor.chunk_documents(org_documents)
            chunked_applicant_docs = text_processor.chunk_documents(applicant_documents)
            
            # Apply rule-based filtering
            rules_config = config.get('rules', {
                'budget': {'max_budget': 1000000},
                'timeline': {'max_duration_months': 24}
            })
            rule_filter = RuleFilter(rules_config)
            filtered_applicants = rule_filter.apply_filters(chunked_applicant_docs)
            
            st.success("Documents processed successfully!")
            
            # Show results
            st.write(f"Organization documents: {len(org_documents)}")
            st.write(f"Applicant documents: {len(applicant_documents)}")
            st.write(f"Chunked organization documents: {len(chunked_org_docs)}")
            st.write(f"Chunked applicant documents: {len(chunked_applicant_docs)}")
            st.write(f"Filtered applicants: {len(filtered_applicants)}")
    
    # Step 3: AI Processing
    st.subheader("3. AI Processing")
    st.info("""
    In a full implementation, this step would:
    - Generate embeddings using Ollama
    - Store in vector database (FAISS/Chroma)
    - Perform semantic similarity search
    - Evaluate with LLM reasoning
    """)
    
    # Step 4: Results
    st.subheader("4. Evaluation Results")
    st.info("""
    In a full implementation, this would display:
    - Ranked list of applicants
    - Compliance scores
    - Detailed analysis
    - Key strengths and weaknesses
    """)
    
    # Step 5: Report Generation
    st.subheader("5. Report Generation")
    st.info("""
    In a full implementation, this would:
    - Generate professional PDF report
    - Include all evaluation results
    - Provide detailed explanations
    - Format as professional document
    """)
    
    if st.button("Generate Sample Report"):
        st.success("Sample report generated successfully!")
        st.download_button(
            label="Download Sample Report (PDF)",
            data="This is a sample report demonstrating the AI-Powered Tender Evaluation System.",
            file_name="tender_evaluation_report.pdf",
            mime="application/pdf"
        )
    
    # System Requirements
    st.header("System Requirements")
    st.markdown("""
    To run the full system, you need:
    - **Ollama**: For local AI processing (embeddings and LLM)
    - **Tesseract OCR**: For processing scanned PDFs
    - **LaTeX**: For generating professional PDF reports
    - **Python dependencies**: As specified in requirements.txt
    
    Install with:
    ```bash
    pip install -r requirements.txt
    ```
    """)


if __name__ == "__main__":
    run_demo_app()