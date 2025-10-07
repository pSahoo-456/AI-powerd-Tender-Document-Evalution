"""
Professional Streamlit interface for the AI-Powered Tender Evaluation System
"""

import streamlit as st
import os
import pandas as pd
import tempfile
from pathlib import Path
from typing import List, Dict, Any

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


def init_session_state():
    """Initialize all session state variables"""
    session_vars = {
        'evaluation_results': None,
        'report_generated': False,
        'report_path': None,
        'org_docs_processed': False,
        'applicant_docs_processed': False,
        'evaluation_completed': False,
        'org_documents': [],
        'applicant_documents': [],
        'org_files': None,
        'applicant_files': None
    }
    
    for key, default_value in session_vars.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def run_professional_app(config_path: str = "./config/config.yaml"):
    """Run the professional Streamlit application"""
    # Set page config
    st.set_page_config(
        page_title="AI-Powered Tender Evaluation System",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2D3748;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .step-card {
        background-color: #F7FAFC;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .result-card {
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .success-box {
        background-color: #D1FAE5;
        border-left: 5px solid #10B981;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
    }
    .warning-box {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
    }
    .error-box {
        background-color: #FEE2E2;
        border-left: 5px solid #EF4444;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
    }
    .progress-bar {
        height: 10px;
        background-color: #E2E8F0;
        border-radius: 5px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    .progress-fill {
        height: 100%;
        background-color: #3B82F6;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header">📄 AI-Powered Tender Evaluation System</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    init_session_state()
    
    # Load configuration
    config_loader = ConfigLoader(config_path)
    config = config_loader.config
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ System Information")
        st.markdown("""
        This system automatically evaluates tender proposals against organization requirements using AI technologies:
        
        - **Document Processing**: PDF text extraction (OCR for scanned docs)
        - **AI Analysis**: Semantic similarity matching with embeddings
        - **Evaluation**: Rule-based filtering + LLM reasoning
        - **Reporting**: Professional LaTeX-generated PDF reports
        """)
        
        st.header("📋 Upload Instructions")
        st.markdown("""
        1. Upload the **Organization Tender Document** (requirements)
        2. Upload **Applicant Proposal Documents** (multiple files)
        3. Click **Process Documents** to extract text
        4. Click **Run Evaluation** to analyze proposals
        5. Download the **Evaluation Report** (PDF)
        """)
        
        st.header("⚙️ System Status")
        if st.session_state.org_docs_processed:
            st.success("✅ Organization docs processed")
        else:
            st.info("📁 Waiting for org docs")
            
        if st.session_state.applicant_docs_processed:
            st.success("✅ Applicant docs processed")
        else:
            st.info("📁 Waiting for applicant docs")
            
        if st.session_state.evaluation_completed:
            st.success("✅ Evaluation completed")
        else:
            st.info("⏳ Waiting for evaluation")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📤 Document Upload", 
        "⚙️ Document Processing", 
        "🧠 AI Evaluation", 
        "📊 Evaluation Results", 
        "📥 Report Generation"
    ])
    
    # Tab 1: Document Upload
    with tab1:
        st.markdown('<h2 class="sub-header">Step 1: Document Upload</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="step-card">', unsafe_allow_html=True)
            st.subheader("🏢 Organization Requirements")
            st.info("Upload the organization's tender document containing the requirements.")
            org_files = st.file_uploader(
                "Upload Tender Requirements (PDF/TXT)",
                type=["pdf", "txt"],
                accept_multiple_files=False,
                key="org_docs"
            )
            st.session_state.org_files = org_files
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="step-card">', unsafe_allow_html=True)
            st.subheader("📋 Applicant Proposals")
            st.info("Upload multiple applicant proposal documents for evaluation.")
            applicant_files = st.file_uploader(
                "Upload Applicant Proposals (PDF/TXT)",
                type=["pdf", "txt"],
                accept_multiple_files=True,
                key="applicant_docs"
            )
            st.session_state.applicant_files = applicant_files
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Store files in session state
        if org_files:
            st.session_state.org_files = org_files
        if applicant_files:
            st.session_state.applicant_files = applicant_files
    
    # Tab 2: Document Processing
    with tab2:
        st.markdown('<h2 class="sub-header">Step 2: Document Processing</h2>', unsafe_allow_html=True)
        
        st.markdown('<div class="step-card">', unsafe_allow_html=True)
        st.subheader("📄 Text Extraction & Processing")
        st.info("Extract text from documents and prepare for AI analysis.")
        
        # Initialize components
        doc_loader = DocumentLoader()
        ocr_processor = OCRProcessor(config.get('ocr', {}))
        text_processor = TextProcessor(config.get('processing', {}))
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔍 Process Organization Document", type="primary"):
                if not st.session_state.get('org_files'):
                    st.warning("Please upload organization documents first.")
                else:
                    with st.spinner("Processing organization document..."):
                        try:
                            st.session_state.org_documents = process_uploaded_file(
                                st.session_state.org_files, 
                                "organization",
                                ocr_processor
                            )
                            st.session_state.org_docs_processed = True
                            st.success(f"✅ Processed {len(st.session_state.org_documents)} organization document chunks")
                        except Exception as e:
                            st.error(f"Error processing organization document: {str(e)}")
        
        with col2:
            if st.button("🔍 Process Applicant Documents", type="primary"):
                if not st.session_state.get('applicant_files'):
                    st.warning("Please upload applicant documents first.")
                else:
                    with st.spinner("Processing applicant documents..."):
                        try:
                            st.session_state.applicant_documents = process_uploaded_files(
                                st.session_state.applicant_files, 
                                "applicant",
                                ocr_processor
                            )
                            st.session_state.applicant_docs_processed = True
                            st.success(f"✅ Processed {len(st.session_state.applicant_documents)} applicant document chunks")
                        except Exception as e:
                            st.error(f"Error processing applicant documents: {str(e)}")
        
        # Show processing status
        if st.session_state.org_docs_processed:
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.markdown("✅ **Organization documents processed successfully**")
            st.markdown('</div>', unsafe_allow_html=True)
        
        if st.session_state.applicant_docs_processed:
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.markdown("✅ **Applicant documents processed successfully**")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 3: AI Evaluation
    with tab3:
        st.markdown('<h2 class="sub-header">Step 3: AI Evaluation</h2>', unsafe_allow_html=True)
        
        st.markdown('<div class="step-card">', unsafe_allow_html=True)
        st.subheader("🤖 AI-Powered Proposal Analysis")
        st.info("Using embeddings and LLM reasoning to evaluate proposals.")
        
        # Evaluation settings
        col1, col2, col3 = st.columns(3)
        
        with col1:
            max_applicants = st.number_input(
                "Maximum Applicants to Evaluate",
                min_value=1,
                max_value=100,
                value=config_loader.get('evaluation.max_applicants', 10),
                help="Number of top applicants to evaluate with LLM"
            )
        
        with col2:
            min_score = st.number_input(
                "Minimum Score Threshold",
                min_value=0,
                max_value=100,
                value=config_loader.get('evaluation_thresholds.minimum_score', 70),
                help="Minimum score for an applicant to be included in results"
            )
        
        with col3:
            st.markdown("")  # Spacer
            st.markdown("")  # Spacer
            if st.button("🚀 Run Full Evaluation", type="primary", use_container_width=True):
                if not st.session_state.org_docs_processed:
                    st.warning("Please process organization documents first.")
                elif not st.session_state.applicant_docs_processed:
                    st.warning("Please process applicant documents first.")
                else:
                    with st.spinner("Running AI evaluation... This may take a few minutes."):
                        try:
                            # Run the full evaluation pipeline
                            evaluation_results = run_full_evaluation(
                                st.session_state.org_documents,
                                st.session_state.applicant_documents,
                                config,
                                max_applicants,
                                min_score
                            )
                            
                            st.session_state.evaluation_results = evaluation_results
                            st.session_state.evaluation_completed = True
                            
                            st.success("✅ AI evaluation completed successfully!")
                            
                            # Show summary
                            st.markdown('<div class="success-box">', unsafe_allow_html=True)
                            st.markdown(f"**Evaluation Summary:**")
                            st.markdown(f"- Qualified Applicants: {len(evaluation_results)}")
                            if evaluation_results:
                                avg_score = sum(result[1].get('score', 0) for result in evaluation_results) / len(evaluation_results)
                                st.markdown(f"- Average Score: {avg_score:.1f}")
                                top_score = max(result[1].get('score', 0) for result in evaluation_results)
                                st.markdown(f"- Top Score: {top_score:.1f}")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.error(f"Error during evaluation: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 4: Evaluation Results
    with tab4:
        st.markdown('<h2 class="sub-header">Step 4: Evaluation Results</h2>', unsafe_allow_html=True)
        
        if not st.session_state.evaluation_completed:
            st.info("Please complete the AI evaluation in the previous step to see results.")
        else:
            st.markdown('<div class="step-card">', unsafe_allow_html=True)
            st.subheader("🏆 Ranked Applicants")
            
            # Convert results to DataFrame for display
            if st.session_state.evaluation_results:
                df_data = []
                for i, (applicant, eval_result) in enumerate(st.session_state.evaluation_results):
                    df_data.append({
                        "Rank": i + 1,
                        "Applicant": applicant.metadata.get('source', f'Applicant {i+1}'),
                        "Similarity Score": eval_result.get('similarity_score', 0),
                        "Compliance Score": eval_result.get('score', 0),
                        "Strengths": eval_result.get('strengths', 'N/A')[:100] + "..." if len(eval_result.get('strengths', 'N/A')) > 100 else eval_result.get('strengths', 'N/A'),
                        "Remarks": eval_result.get('explanation', 'N/A')[:100] + "..." if len(eval_result.get('explanation', 'N/A')) > 100 else eval_result.get('explanation', 'N/A')
                    })
                
                df = pd.DataFrame(df_data)
                
                # Display interactive table
                st.dataframe(df, use_container_width=True, height=400)
                
                # Detailed view
                st.subheader("📋 Detailed Analysis")
                for i, (applicant, eval_result) in enumerate(st.session_state.evaluation_results):
                    with st.expander(f"#{i+1} - {applicant.metadata.get('source', f'Applicant {i+1}')}", expanded=i==0):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"**Compliance Score:** {eval_result.get('score', 0)}")
                            st.progress(eval_result.get('score', 0) / 100)
                            st.markdown(f"**Similarity Score:** {eval_result.get('similarity_score', 0):.3f}")
                        
                        with col2:
                            st.markdown(f"**Strengths:**")
                            st.markdown(eval_result.get('strengths', 'N/A'))
                            st.markdown(f"**Areas for Improvement:**")
                            st.markdown(eval_result.get('improvements', 'N/A'))
                        
                        st.markdown(f"**Explanation:**")
                        st.markdown(eval_result.get('explanation', 'N/A'))
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 5: Report Generation
    with tab5:
        st.markdown('<h2 class="sub-header">Step 5: Report Generation</h2>', unsafe_allow_html=True)
        
        st.markdown('<div class="step-card">', unsafe_allow_html=True)
        st.subheader("📑 Professional PDF Report")
        st.info("Generate a comprehensive LaTeX-formatted evaluation report.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            report_title = st.text_input(
                "Report Title",
                "Tender Evaluation Report",
                help="Title for the evaluation report"
            )
            
            if st.button("📄 Generate LaTeX Report", type="primary"):
                if not st.session_state.evaluation_completed:
                    st.warning("Please complete the AI evaluation first.")
                else:
                    with st.spinner("Generating LaTeX report..."):
                        try:
                            report_generator = ReportGenerator()
                            report_path = report_generator.generate_evaluation_report(
                                st.session_state.org_documents,
                                st.session_state.evaluation_results,
                                report_title
                            )
                            
                            st.session_state.report_path = report_path
                            st.session_state.report_generated = True
                            
                            st.success("✅ LaTeX report generated successfully!")
                            st.info(f"Report saved to: {report_path}")
                        except Exception as e:
                            st.error(f"Error generating report: {str(e)}")
        
        with col2:
            if st.session_state.report_generated and st.session_state.report_path:
                st.subheader("📥 Download Report")
                if os.path.exists(st.session_state.report_path):
                    with open(st.session_state.report_path, "rb") as file:
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=file,
                            file_name=os.path.basename(st.session_state.report_path),
                            mime="application/pdf",
                            use_container_width=True
                        )
                else:
                    st.warning("Report file not found.")
        
        st.markdown('</div>', unsafe_allow_html=True)


def process_uploaded_file(uploaded_file, source_type: str, ocr_processor: OCRProcessor) -> List:
    """Process a single uploaded file and convert to documents"""
    documents = []
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        temp_path = tmp_file.name
    
    try:
        # Process based on file type
        if uploaded_file.name.endswith('.pdf'):
            # For PDFs, check if OCR is needed
            if ocr_processor.is_scanned_pdf(temp_path):
                # Use OCR for scanned PDFs
                extracted_text = ocr_processor.process_scanned_pdf(temp_path)
                from langchain_core.documents import Document
                doc = Document(
                    page_content=extracted_text,
                    metadata={"source": uploaded_file.name, "type": source_type}
                )
                documents.append(doc)
            else:
                # Use regular PDF loader for digital PDFs
                doc_loader = DocumentLoader()
                pdf_docs = doc_loader._load_pdf_document(temp_path)
                for doc in pdf_docs:
                    doc.metadata["source"] = uploaded_file.name
                    doc.metadata["type"] = source_type
                documents.extend(pdf_docs)
        elif uploaded_file.name.endswith('.txt'):
            doc_loader = DocumentLoader()
            txt_docs = doc_loader._load_text_document(temp_path)
            for doc in txt_docs:
                doc.metadata["source"] = uploaded_file.name
                doc.metadata["type"] = source_type
            documents.extend(txt_docs)
    finally:
        # Clean up temporary file
        os.unlink(temp_path)
    
    return documents


def process_uploaded_files(uploaded_files, source_type: str, ocr_processor: OCRProcessor) -> List:
    """Process multiple uploaded files and convert to documents"""
    all_documents = []
    
    for uploaded_file in uploaded_files:
        try:
            documents = process_uploaded_file(uploaded_file, source_type, ocr_processor)
            all_documents.extend(documents)
        except Exception as e:
            st.warning(f"Could not process {uploaded_file.name}: {str(e)}")
    
    return all_documents


def run_full_evaluation(org_documents: List, applicant_documents: List, 
                       config: Dict[str, Any], max_applicants: int, min_score: int) -> List:
    """Run the full evaluation pipeline"""
    try:
        # Initialize components
        text_processor = TextProcessor(config.get('processing', {}))
        embedding_generator = EmbeddingGenerator(config.get('ollama', {}))
        vector_store_manager = VectorStoreManager(config.get('vector_db', {}))
        similarity_searcher = SimilaritySearcher(vector_store_manager)
        rule_filter = RuleFilter(config.get('rules', {}))
        llm_evaluator = LLMEvaluator(config.get('ollama', {}))
        
        # Process documents
        chunked_org_docs = text_processor.chunk_documents(org_documents)
        chunked_applicant_docs = text_processor.chunk_documents(applicant_documents)
        
        # Generate embeddings
        try:
            embedded_org_docs = embedding_generator.generate_document_embeddings(chunked_org_docs)
            embedded_applicant_docs = embedding_generator.generate_document_embeddings(chunked_applicant_docs)
        except Exception as e:
            st.error(f"Embedding generation failed: {str(e)}")
            if "not found" in str(e) or "404" in str(e):
                st.info("💡 Tip: To fix this, run the following command in your terminal:")
                st.code(f"ollama pull {config.get('ollama', {}).get('embedding_model', 'nomic-embed-text')}")
                st.info("Or continue with simulated embeddings by clicking 'Continue Anyway' below.")
                
                # Add a button to continue with simulated embeddings
                if st.button("Continue Anyway (Use Simulated Embeddings)"):
                    # Use simulated embeddings
                    embedded_org_docs = embedding_generator.generate_document_embeddings(chunked_org_docs)
                    embedded_applicant_docs = embedding_generator.generate_document_embeddings(chunked_applicant_docs)
                else:
                    return []
            else:
                st.error("An unexpected error occurred during embedding generation.")
                return []
        
        # Initialize vector store
        vector_store_manager.initialize_store(embedded_org_docs + embedded_applicant_docs)
        
        # Apply rule-based filtering
        filtered_applicants = rule_filter.apply_filters(embedded_applicant_docs)
        
        # Perform similarity search
        similar_applicants = similarity_searcher.search_applicants_by_requirements(
            embedded_org_docs, filtered_applicants, top_k=max_applicants
        )
        
        # Add similarity scores to evaluation results
        scored_applicants = []
        for doc, score in similar_applicants:
            # Create a copy of the document with similarity score in metadata
            from langchain_core.documents import Document
            scored_doc = Document(
                page_content=doc.page_content,
                metadata=doc.metadata.copy() if doc.metadata else {}
            )
            scored_doc.metadata['similarity_score'] = score
            scored_applicants.append(scored_doc)
        
        # Evaluate with LLM
        evaluated_applicants = llm_evaluator.evaluate_applicants(
            embedded_org_docs, 
            scored_applicants,
            max_applicants=max_applicants
        )
        
        # Filter by minimum score
        qualified_applicants = [
            (doc, eval_result) for doc, eval_result in evaluated_applicants
            if eval_result.get('score', 0) >= min_score
        ]
        
        return qualified_applicants
        
    except Exception as e:
        st.error(f"Error during evaluation: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return []


if __name__ == "__main__":
    run_professional_app()