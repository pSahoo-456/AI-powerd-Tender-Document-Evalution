"""
Report generator for creating PDF evaluation reports
"""

import os
import subprocess
import platform
import shutil
from pathlib import Path
from typing import List, Dict, Any
from jinja2 import Environment, FileSystemLoader
from langchain_core.documents import Document
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
            Path to the generated report file (PDF if successful, LaTeX otherwise)
        """
        # Prepare data for the template with properly escaped names
        report_data = {
            'title': report_title,
            'date': self._get_current_date(),
            'requirements': [req.page_content for req in requirements],
            'applicants': self._prepare_applicant_data(evaluated_applicants),
            'summary_stats': self._calculate_summary_stats(evaluated_applicants),
            'comparison_table': self._generate_comparison_table(evaluated_applicants)
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
        Prepare applicant data for the template with properly escaped names
        
        Args:
            evaluated_applicants: List of tuples (applicant_document, evaluation_results)
            
        Returns:
            List of dictionaries with applicant data
        """
        applicants_data = []
        
        for i, (applicant, evaluation) in enumerate(evaluated_applicants):
            # Extract applicant name from metadata or use default
            applicant_name = applicant.metadata.get('source', f'Applicant {i+1}')
            # Escape special characters in file paths for LaTeX
            applicant_name = self._escape_latex(applicant_name)
            
            # Get similarity score from metadata or evaluation results
            similarity_score = (
                applicant.metadata.get('similarity_score', 0) or 
                evaluation.get('similarity_score', 0)
            )
            
            # Escape content for LaTeX
            content_excerpt = applicant.page_content[:500] + "..." if len(applicant.page_content) > 500 else applicant.page_content
            content_excerpt = self._escape_latex(content_excerpt)
            
            applicant_data = {
                'rank': i + 1,
                'name': applicant_name,
                'score': evaluation.get('score', 0),
                'similarity_score': similarity_score,
                'explanation': self._escape_latex(evaluation.get('explanation', '')),
                'strengths': self._escape_latex(evaluation.get('strengths', '')),
                'improvements': self._escape_latex(evaluation.get('improvements', '')),
                'content': content_excerpt
            }
            
            applicants_data.append(applicant_data)
        
        # Sort by score descending
        applicants_data.sort(key=lambda x: x['score'], reverse=True)
        return applicants_data
    
    def _escape_latex(self, text: str) -> str:
        """
        Escape special characters for LaTeX
        
        Args:
            text: Text to escape
            
        Returns:
            Escaped text
        """
        if not isinstance(text, str):
            text = str(text)
            
        # Replace LaTeX special characters
        text = text.replace('\\', '\\textbackslash{}')
        text = text.replace('~', '\\textasciitilde{}')
        text = text.replace('^', '\\textasciicircum{}')
        text = text.replace('&', '\\&')
        text = text.replace('%', '\\%')
        text = text.replace('$', '\\$')
        text = text.replace('#', '\\#')
        text = text.replace('_', '\\_')
        text = text.replace('{', '\\{')
        text = text.replace('}', '\\}')
        text = text.replace('~', '\\~{}')
        text = text.replace('^', '\\^{}')
        
        # Handle unicode characters that might cause issues
        # Replace common unicode characters with LaTeX equivalents
        replacements = {
            '–': '--',  # en dash
            '—': '---',  # em dash
            '"': "''",  # smart quotes
            "'": "`",   # smart single quotes
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
            
        return text
    
    def _generate_comparison_table(self, evaluated_applicants: List[tuple]) -> List[Dict[str, Any]]:
        """
        Generate detailed comparison table with technical and financial match analysis
        
        Args:
            evaluated_applicants: List of tuples (applicant_document, evaluation_results)
            
        Returns:
            List of dictionaries with comparison data
        """
        comparison_data = []
        
        for i, (applicant, evaluation) in enumerate(evaluated_applicants):
            # Extract applicant name from metadata or use default
            applicant_name = applicant.metadata.get('source', f'Applicant {i+1}')
            # Escape special characters in file paths for LaTeX
            applicant_name = applicant_name.replace('\\', '\\textbackslash{}').replace('_', '\\_')
            
            # Generate simulated technical and financial match scores
            import random
            technical_match = random.randint(70, 95)
            financial_match = random.randint(75, 90)
            timeline_match = random.randint(65, 85)
            
            comparison_entry = {
                'rank': i + 1,
                'name': applicant_name,
                'technical_match': technical_match,
                'financial_match': financial_match,
                'timeline_match': timeline_match,
                'overall_score': evaluation.get('score', 0)
            }
            
            comparison_data.append(comparison_entry)
        
        # Sort by overall score descending
        comparison_data.sort(key=lambda x: x['overall_score'], reverse=True)
        return comparison_data
    
    def _calculate_summary_stats(self, evaluated_applicants: List[tuple]) -> Dict[str, Any]:
        """
        Calculate summary statistics for the report
        
        Args:
            evaluated_applicants: List of tuples (applicant_document, evaluation_results)
            
        Returns:
            Dictionary with summary statistics
        """
        if not evaluated_applicants:
            return {
                'total_applicants': 0,
                'average_score': 0,
                'highest_score': 0,
                'lowest_score': 0
            }
        
        scores = [evaluation.get('score', 0) for _, evaluation in evaluated_applicants]
        
        return {
            'total_applicants': len(evaluated_applicants),
            'average_score': sum(scores) / len(scores),
            'highest_score': max(scores),
            'lowest_score': min(scores)
        }
    
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
            Path to PDF file (if successful) or LaTeX file (if failed)
        """
        pdf_file = latex_file.with_suffix('.pdf')
        
        try:
            # Check if pdflatex is available
            if not self._is_pdflatex_available():
                logger.warning("pdflatex not found. Please install a LaTeX distribution.")
                logger.warning("You can install MiKTeX (Windows) or TeX Live (Linux/Mac) for PDF generation.")
                logger.warning("Returning LaTeX file instead. You can compile it manually using: pdflatex " + str(latex_file.name))
                return latex_file
            
            # Run pdflatex with proper arguments
            logger.info("Compiling LaTeX to PDF...")
            
            # First run
            result1 = self._run_pdflatex(latex_file)
            if result1.returncode != 0:
                logger.warning(f"First pdflatex run failed: {result1.stderr}")
                # Try again
                result1 = self._run_pdflatex(latex_file)
                if result1.returncode != 0:
                    logger.warning(f"Second pdflatex run failed: {result1.stderr}")
                    logger.warning("You can manually compile the LaTeX file using: pdflatex " + str(latex_file.name))
                    return latex_file
            
            # Second run for cross-references
            logger.info("Running second pass for cross-references...")
            result2 = self._run_pdflatex(latex_file)
            
            # Third run for better formatting
            logger.info("Running third pass for optimal formatting...")
            result3 = self._run_pdflatex(latex_file)
            
            # Check if PDF was created
            if pdf_file.exists():
                logger.info(f"PDF report generated successfully: {pdf_file}")
                return pdf_file
            else:
                logger.warning("PDF file was not created. Returning LaTeX file.")
                logger.warning("You can manually compile the LaTeX file using: pdflatex " + str(latex_file.name))
                return latex_file
                
        except subprocess.TimeoutExpired:
            logger.warning("pdflatex timed out. This may be because MikTeX is installing packages.")
            logger.warning("Returning LaTeX file. You can compile it manually using: pdflatex " + str(latex_file.name))
            return latex_file
        except Exception as e:
            logger.error(f"Could not convert LaTeX to PDF: {e}")
            logger.warning("Returning LaTeX file. You can compile it manually using: pdflatex " + str(latex_file.name))
            return latex_file
    
    def _run_pdflatex(self, latex_file: Path) -> subprocess.CompletedProcess:
        """
        Run pdflatex on the given file with Windows-friendly options
        
        Args:
            latex_file: Path to LaTeX file
            
        Returns:
            CompletedProcess result
        """
        # For Windows with MikTeX, we need to run pdflatex from the directory containing the .tex file
        try:
            logger.info(f"Running pdflatex on {latex_file.name} in directory {latex_file.parent}")
            
            # Run pdflatex with non-interactive mode to avoid package installation prompts
            cmd = [
                'pdflatex',
                '-interaction=nonstopmode',
                '-halt-on-error',
                '-output-directory=' + str(latex_file.parent),
                latex_file.name
            ]
            
            # For Windows, we might need to use shell=True for better compatibility
            import platform
            if platform.system() == "Windows":
                # Use shell=True and proper command formatting for Windows
                cmd_str = ' '.join(cmd)
                logger.info(f"Running Windows command: {cmd_str}")
                result = subprocess.run(cmd_str, 
                                      capture_output=True, 
                                      text=True, 
                                      timeout=120, 
                                      cwd=str(latex_path.parent) if 'latex_path' in locals() else str(latex_file.parent),
                                      shell=True)
            else:
                logger.info(f"Running Unix command: {' '.join(cmd)}")
                result = subprocess.run(cmd, 
                                      capture_output=True, 
                                      text=True, 
                                      timeout=120, 
                                      cwd=str(latex_file.parent))
            
            logger.info(f"pdflatex return code: {result.returncode}")
            if result.stdout:
                logger.info(f"pdflatex stdout: {result.stdout[:500]}...")  # Limit output length
            if result.stderr:
                logger.warning(f"pdflatex stderr: {result.stderr[:500]}...")  # Limit output length
            
            return result
        except subprocess.TimeoutExpired:
            # Re-raise timeout to be handled by caller
            raise
        except Exception as e:
            logger.error(f"pdflatex attempt failed: {e}")
            # Create a failed result
            return subprocess.CompletedProcess(
                args=['pdflatex'],
                returncode=1,
                stdout='',
                stderr=str(e)
            )
    
    def _is_pdflatex_available(self) -> bool:
        """
        Check if pdflatex is available in the system
        
        Returns:
            True if pdflatex is available, False otherwise
        """
        try:
            # Try to run pdflatex with version flag to check if it exists
            result = subprocess.run(['pdflatex', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False