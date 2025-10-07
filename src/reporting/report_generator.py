"""
Report generator for creating PDF evaluation reports
"""

import os
from pathlib import Path
from typing import List, Dict, Any
from jinja2 import Environment, FileSystemLoader
from langchain_core.documents import Document


class ReportGenerator:
    """Generate structured reports using Jinja2 templates"""
    
    def __init__(self, template_dir: str = "./templates", output_dir: str = "./data/reports"):
        """
        Initialize report generator
        
        Args:
            template_dir: Directory containing Jinja2 templates
            output_dir: Directory to save generated reports
        """
        self.template_dir = Path(template_dir)
        self.output_dir = Path(output_dir)
        
        # Ensure directories exist
        self.template_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Jinja2 environment
        self.env = Environment(loader=FileSystemLoader(str(self.template_dir)))
    
    def generate_evaluation_report(self, requirements: List[Document], 
                                 evaluated_applicants: List[tuple],
                                 report_title: str = "Tender Evaluation Report") -> str:
        """
        Generate a PDF report of the evaluation results
        
        Args:
            requirements: List of requirement Document objects
            evaluated_applicants: List of tuples (applicant_document, evaluation_results)
            report_title: Title for the report
            
        Returns:
            Path to the generated report file
        """
        # Prepare data for the template
        report_data = {
            'title': report_title,
            'date': self._get_current_date(),
            'requirements': [req.page_content for req in requirements],
            'applicants': self._prepare_applicant_data(evaluated_applicants)
        }
        
        # Render the LaTeX template
        template = self.env.get_template('report_template.tex')
        latex_content = template.render(**report_data)
        
        # Save LaTeX file
        latex_file = self.output_dir / f"{report_title.replace(' ', '_')}.tex"
        with open(latex_file, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        
        # Convert to PDF (if LaTeX is installed)
        pdf_file = self._convert_to_pdf(latex_file)
        
        return str(pdf_file)
    
    def _prepare_applicant_data(self, evaluated_applicants: List[tuple]) -> List[Dict[str, Any]]:
        """
        Prepare applicant data for the report template
        
        Args:
            evaluated_applicants: List of tuples (applicant_document, evaluation_results)
            
        Returns:
            List of dictionaries with applicant data
        """
        applicants_data = []
        
        for i, (applicant, evaluation) in enumerate(evaluated_applicants):
            # Extract applicant name from metadata or use default
            applicant_name = applicant.metadata.get('source', f'Applicant {i+1}')
            
            # Get similarity score from metadata or evaluation results
            similarity_score = (
                applicant.metadata.get('similarity_score', 0) or 
                evaluation.get('similarity_score', 0)
            )
            
            applicant_data = {
                'rank': i + 1,
                'name': applicant_name,
                'score': evaluation.get('score', 0),
                'similarity_score': similarity_score,
                'explanation': evaluation.get('explanation', ''),
                'strengths': evaluation.get('strengths', ''),
                'improvements': evaluation.get('improvements', ''),
                'content': applicant.page_content[:500] + "..." if len(applicant.page_content) > 500 else applicant.page_content
            }
            
            applicants_data.append(applicant_data)
        
        return applicants_data
    
    def _get_current_date(self) -> str:
        """Get current date as string"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d")
    
    def _convert_to_pdf(self, latex_file: Path) -> Path:
        """
        Convert LaTeX file to PDF
        
        Args:
            latex_file: Path to LaTeX file
            
        Returns:
            Path to PDF file
        """
        pdf_file = latex_file.with_suffix('.pdf')
        
        try:
            # Try to compile LaTeX to PDF using pdflatex
            import subprocess
            result = subprocess.run([
                'pdflatex', 
                '-output-directory', str(self.output_dir),
                str(latex_file)
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                return pdf_file
            else:
                print(f"Warning: Failed to compile LaTeX to PDF: {result.stderr}")
        except Exception as e:
            print(f"Warning: Could not convert LaTeX to PDF: {e}")
        
        # Return LaTeX file if PDF conversion failed
        return latex_file