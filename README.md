# AI-Powered Tender Evaluation System

This system automates the evaluation of tender documents against applicants' proposals using AI technologies.

## Features
- Parse organization tender documents to extract requirements
- Process applicants' PDF documents (OCR for scanned docs)
- Generate embeddings using Ollama
- Perform semantic similarity search
- Apply rule-based filters
- Evaluate top candidates with LLM
- Generate professional PDF reports

## Tech Stack
- LangChain for pipeline orchestration
- Tesseract OCR for text extraction
- Ollama for embeddings and LLM reasoning
- FAISS/Chroma as vector database
- Jinja2 + LaTeX for report generation
- Streamlit for UI

## Project Structure
```
tender-evaluation-system/
├── config/
│   ├── config.yaml
│   └── tender_rules.yaml
├── data/
│   ├── org_documents/
│   └── applicant_documents/
├── src/
│   ├── ingestion/
│   ├── ocr/
│   ├── parsing/
│   ├── embeddings/
│   ├── vector_db/
│   ├── search/
│   ├── filtering/
│   ├── evaluation/
│   ├── reporting/
│   ├── interfaces/
│   └── utils/
├── templates/
├── tests/
├── requirements.txt
├── main.py
└── README.md
```

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Install Ollama from https://ollama.ai and pull the required models:
```bash
ollama pull nomic-embed-text
ollama pull llama3.1
```

3. Install Tesseract OCR:
   - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
   - macOS: `brew install tesseract`
   - Linux: `sudo apt-get install tesseract-ocr`

4. Install LaTeX distribution for report generation:
   - Windows: Install MiKTeX or TeX Live
   - macOS: Install MacTeX
   - Linux: `sudo apt-get install texlive-full`

## Usage

### CLI Interface
```bash
python main.py --mode cli
```

### Professional Web Interface (Streamlit)
```bash
python main.py --mode web
```

### Demo Interface (Streamlit)
```bash
python main.py --mode demo
```

### Adding Documents
1. Place organization tender documents in `data/org_documents/`
2. Place applicant proposals in `data/applicant_documents/`
3. Supported formats: PDF, TXT

## Configuration

The system can be configured using `config/config.yaml`:
- Ollama settings (model names, base URL)
- Vector database settings (FAISS or Chroma)
- Processing parameters (chunk size, overlap)
- Evaluation settings (top K results, scoring weights)

Tender rules are configured in `config/tender_rules.yaml`:
- Budget constraints
- Timeline requirements
- Certification requirements
- Technical requirements
- Scoring weights

## System Workflow

1. **Document Ingestion**: Load organization and applicant documents
2. **OCR Processing**: Extract text from scanned PDFs if needed
3. **Text Parsing**: Chunk documents into manageable pieces
4. **Embedding Generation**: Create vector representations using Ollama
5. **Vector Storage**: Store embeddings in FAISS or Chroma database
6. **Similarity Search**: Find semantically similar applicant documents
7. **Rule Filtering**: Apply budget, timeline, and certification filters
8. **LLM Evaluation**: Use LLM to score and analyze top candidates
9. **Report Generation**: Create professional PDF reports with results

## Professional Web Interface Features

The professional Streamlit interface provides:

- **Modern UI**: Clean, professional design with intuitive navigation
- **Tabbed Workflow**: Organized 5-step process for document evaluation
- **Document Upload**: Separate sections for organization requirements and applicant proposals
- **AI Processing**: Real-time status updates during document processing
- **Interactive Results**: Sortable tables with detailed applicant analysis
- **PDF Reports**: Professional LaTeX-generated reports with comparison tables
- **Progress Indicators**: Visual feedback throughout the evaluation process

## Testing the System

### Running Component Tests
```bash
python -m pytest tests/ -v
```

### Running Demonstration
```bash
python demonstration.py
```

### Running Professional Web Interface
```bash
streamlit run main.py -- --mode web
```

### Running Demo Interface
```bash
streamlit run main.py -- --mode demo
```

## License

This project is licensed under the MIT License.