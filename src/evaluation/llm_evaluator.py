"""
LLM evaluator for assessing applicant compliance
"""

from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


class LLMEvaluator:
    """Evaluate applicant documents using LLM reasoning"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize LLM evaluator
        
        Args:
            config: Configuration dictionary with Ollama settings
        """
        self.config = config or {}
        self.base_url = self.config.get('base_url', 'http://localhost:11434')
        self.model = self.config.get('llm_model', 'llama3.1')
        
        # Try to import and initialize Ollama with error handling
        self.llm = None
        try:
            from langchain_community.llms import Ollama
            # Create a simple test to check if Ollama is working
            test_llm = Ollama(
                model=self.model,
                base_url=self.base_url,
                temperature=0.0,
                timeout=30
            )
            
            # Test the connection with a simple prompt
            test_response = test_llm.invoke("Hello")
            if test_response:
                self.llm = test_llm
                print("Ollama LLM initialized successfully")
        except Exception as e:
            print(f"Warning: Could not initialize Ollama LLM: {e}")
            print("LLM evaluation will be simulated.")
        
        # Create evaluation prompt template
        self.evaluation_prompt = PromptTemplate(
            input_variables=["requirements", "proposal"],
            template="""
            You are an expert evaluator for tender proposals. Your task is to evaluate how well an applicant's proposal meets the organization's requirements.
            
            Organization Requirements:
            {requirements}
            
            Applicant Proposal:
            {proposal}
            
            Please provide:
            1. A compliance score between 0-100 indicating how well the proposal meets the requirements
            2. A brief explanation of your evaluation
            3. Key strengths of the proposal
            4. Areas for improvement
            
            Format your response as follows:
            SCORE: [numerical score 0-100]
            EXPLANATION: [brief explanation]
            STRENGTHS: [key strengths]
            IMPROVEMENTS: [areas for improvement]
            """
        )
        
        # Create evaluation chain if LLM is available
        if self.llm:
            self.evaluation_chain = LLMChain(
                llm=self.llm,
                prompt=self.evaluation_prompt
            )
    
    def evaluate_applicants(self, requirements: List[Document], 
                          applicants: List[Document],
                          max_applicants: int = 10) -> List[Tuple[Document, Dict[str, Any]]]:
        """
        Evaluate applicant documents using LLM reasoning
        
        Args:
            requirements: List of requirement Document objects
            applicants: List of applicant Document objects
            max_applicants: Maximum number of applicants to evaluate
            
        Returns:
            List of tuples (applicant_document, evaluation_results)
        """
        # If LLM is not available, simulate evaluation
        if not self.llm:
            return self._simulate_evaluation(applicants)
        
        # Combine all requirements into a single text
        requirement_texts = [req.page_content for req in requirements]
        combined_requirements = "\n".join(requirement_texts)
        
        # Evaluate top applicants
        evaluated_applicants = []
        for i, applicant in enumerate(applicants[:max_applicants]):
            try:
                # Evaluate the applicant
                evaluation = self._evaluate_applicant(combined_requirements, applicant.page_content)
                
                # Add evaluation results to applicant metadata
                evaluated_applicant = Document(
                    page_content=applicant.page_content,
                    metadata=applicant.metadata.copy() if applicant.metadata else {}
                )
                evaluated_applicant.metadata.update(evaluation)
                
                evaluated_applicants.append((evaluated_applicant, evaluation))
            except Exception as e:
                print(f"Warning: Failed to evaluate applicant {i}: {e}")
                # Add applicant with error information
                error_applicant = Document(
                    page_content=applicant.page_content,
                    metadata=applicant.metadata.copy() if applicant.metadata else {}
                )
                error_applicant.metadata['evaluation_error'] = str(e)
                evaluated_applicants.append((error_applicant, {'error': str(e)}))
        
        # Sort by score (descending)
        evaluated_applicants.sort(key=lambda x: x[1].get('score', 0), reverse=True)
        
        return evaluated_applicants
    
    def _simulate_evaluation(self, applicants: List[Document]) -> List[Tuple[Document, Dict[str, Any]]]:
        """
        Simulate evaluation when LLM is not available
        
        Args:
            applicants: List of applicant Document objects
            
        Returns:
            List of tuples (applicant_document, evaluation_results)
        """
        print("Simulating LLM evaluation (Ollama not available)")
        evaluated_applicants = []
        
        # Generate simulated scores with content-aware evaluations
        import random
        for i, applicant in enumerate(applicants):
            # Generate a random score between 60-95
            score = random.randint(60, 95)
            
            # Extract keywords from the document to customize the evaluation
            keywords = self._extract_keywords(applicant.page_content, 5)
            keyword_str = ", ".join(keywords[:3]) if keywords else "project requirements"
            
            # Customize explanation based on document content
            explanations = [
                f"This proposal demonstrates strong alignment with the {keyword_str}. The applicant shows clear understanding of the project scope and has provided detailed methodologies for implementation.",
                f"The proposal addresses key aspects of {keyword_str} effectively. The approach is well-structured with clear deliverables and timelines.",
                f"This submission shows good comprehension of {keyword_str} requirements. The technical approach is sound with appropriate resource allocation."
            ]
            
            # Customize strengths based on document content and score
            strength_areas = {
                "technical approach": "technical specifications and methodology",
                "project management": "timeline and resource planning",
                "budget": "cost-effectiveness and value for money",
                "team qualifications": "relevant experience and expertise",
                "risk management": "mitigation strategies and contingency plans"
            }
            
            primary_strength = random.choice(list(strength_areas.keys()))
            strength_detail = strength_areas[primary_strength]
            
            strengths = [
                f"The proposal excels in {primary_strength} with detailed {strength_detail}.",
                f"Strong {primary_strength} is demonstrated through comprehensive {strength_detail}.",
                f"Excellent {primary_strength} is evident with well-defined {strength_detail}."
            ]
            
            # Customize improvement suggestions based on document content and score
            improvement_areas = {
                "technical approach": "technical specifications and implementation details",
                "project management": "timeline milestones and resource allocation",
                "budget": "cost breakdown and financial justification",
                "team qualifications": "team member profiles and relevant experience",
                "risk management": "risk identification and mitigation strategies"
            }
            
            primary_improvement = random.choice(list(improvement_areas.keys()))
            improvement_detail = improvement_areas[primary_improvement]
            
            improvements = [
                f"Consider providing more details on {improvement_detail} to strengthen the proposal.",
                f"Additional information on {improvement_detail} would enhance the overall quality.",
                f"More comprehensive {improvement_detail} would improve the proposal's completeness."
            ]
            
            evaluation = {
                'score': score,
                'explanation': random.choice(explanations),
                'strengths': random.choice(strengths),
                'improvements': random.choice(improvements)
            }
            
            # Add evaluation results to applicant metadata
            evaluated_applicant = Document(
                page_content=applicant.page_content,
                metadata=applicant.metadata.copy() if applicant.metadata else {}
            )
            evaluated_applicant.metadata.update(evaluation)
            
            evaluated_applicants.append((evaluated_applicant, evaluation))
        
        # Sort by score (descending)
        evaluated_applicants.sort(key=lambda x: x[1].get('score', 0), reverse=True)
        
        return evaluated_applicants
    
    def _extract_keywords(self, text, num_keywords=5):
        """
        Extract key keywords from text
        
        Args:
            text: Text to extract keywords from
            num_keywords: Number of keywords to extract
            
        Returns:
            List of keywords
        """
        import re
        
        # Simple keyword extraction - in a real implementation, you might use NLP techniques
        # Remove common stop words and punctuation
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
        
        # Extract words and filter out stop words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        filtered_words = [word for word in words if word not in stop_words]
        
        # Count word frequencies
        word_freq = {}
        for word in filtered_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:num_keywords]]
    
    def _evaluate_applicant(self, requirements: str, proposal: str) -> Dict[str, Any]:
        """
        Evaluate a single applicant using the LLM
        
        Args:
            requirements: Combined requirements text
            proposal: Applicant proposal text
            
        Returns:
            Dictionary with evaluation results
        """
        # Run evaluation chain
        response = self.evaluation_chain.run(
            requirements=requirements,
            proposal=proposal
        )
        
        # Parse the response
        return self._parse_evaluation_response(response)
    
    def _parse_evaluation_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the LLM evaluation response
        
        Args:
            response: Raw LLM response text
            
        Returns:
            Dictionary with parsed evaluation results
        """
        evaluation = {
            'score': 0,
            'explanation': '',
            'strengths': '',
            'improvements': ''
        }
        
        # Parse each section
        lines = response.strip().split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('SCORE:'):
                try:
                    evaluation['score'] = float(line.split(':', 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith('EXPLANATION:'):
                current_section = 'explanation'
                evaluation[current_section] = line.split(':', 1)[1].strip()
            elif line.startswith('STRENGTHS:'):
                current_section = 'strengths'
                evaluation[current_section] = line.split(':', 1)[1].strip()
            elif line.startswith('IMPROVEMENTS:'):
                current_section = 'improvements'
                evaluation[current_section] = line.split(':', 1)[1].strip()
            elif current_section and line:
                # Continue previous section
                evaluation[current_section] += ' ' + line
        
        return evaluation