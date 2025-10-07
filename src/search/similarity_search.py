"""
Similarity search functionality for document comparison
"""

from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from src.vector_db.vector_store import VectorStoreManager


class SimilaritySearcher:
    """Perform similarity search between organization requirements and applicant documents"""
    
    def __init__(self, vector_store_manager: VectorStoreManager):
        """
        Initialize similarity searcher
        
        Args:
            vector_store_manager: Vector store manager instance
        """
        self.vector_store = vector_store_manager
    
    def search_applicants_by_requirements(self, requirements: List[Document], 
                                        applicants: List[Document], 
                                        top_k: int = 10) -> List[Tuple[Document, float]]:
        """
        Search for the most similar applicant documents to the organization requirements
        
        Args:
            requirements: List of requirement documents
            applicants: List of applicant documents
            top_k: Number of top matches to return
            
        Returns:
            List of tuples (applicant_document, similarity_score)
        """
        # Combine all requirement texts into a single query
        requirement_texts = [req.page_content for req in requirements]
        combined_query = " ".join(requirement_texts)
        
        # Perform similarity search
        # Note: The FAISS/Chroma implementation should return documents with similarity scores
        # For now, we'll simulate this by calculating our own scores
        similar_docs = self.vector_store.similarity_search(combined_query, k=top_k)
        
        # Calculate similarity scores for each document
        results = []
        for doc in similar_docs:
            # In a real implementation with FAISS/Chroma, scores would be available
            # For now, we'll use a placeholder score or extract from metadata if available
            score = getattr(doc, 'similarity_score', self._calculate_similarity(combined_query, doc.page_content))
            results.append((doc, score))
        
        return results
    
    def rank_applicants(self, requirements: List[Document], 
                       applicants: List[Document]) -> List[Tuple[Document, float]]:
        """
        Rank applicants based on similarity to requirements
        
        Args:
            requirements: List of requirement documents
            applicants: List of applicant documents
            
        Returns:
            List of tuples (applicant_document, similarity_score) sorted by score
        """
        # This is a simplified implementation
        # In a real system, you might want to use more sophisticated ranking algorithms
        
        # For each applicant, calculate similarity to requirements
        ranked_applicants = []
        
        # Combine all requirement texts into a single query
        requirement_texts = [req.page_content for req in requirements]
        combined_query = " ".join(requirement_texts)
        
        # Search for each applicant document
        for applicant in applicants:
            # In a real implementation, we would calculate actual similarity scores
            # For now, we'll use a placeholder approach
            score = self._calculate_similarity(combined_query, applicant.page_content)
            ranked_applicants.append((applicant, score))
        
        # Sort by similarity score (descending)
        ranked_applicants.sort(key=lambda x: x[1], reverse=True)
        
        return ranked_applicants
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts (placeholder implementation)
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        # This is a very simplified similarity calculation
        # In a real implementation, you would use the actual vector similarities
        
        # Convert to lowercase and split into words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Calculate Jaccard similarity
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        if len(union) == 0:
            return 0.0
        
        return len(intersection) / len(union)