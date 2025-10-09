"""
Similarity search functionality for document comparison
"""

from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from src.vector_db.vector_store import VectorStoreManager
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


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
        try:
            similar_docs = self.vector_store.similarity_search(combined_query, k=top_k * 2)  # Get more results to filter
        except Exception as e:
            # If similarity search fails, fall back to ranking all applicants
            print(f"Similarity search failed, falling back to ranking: {e}")
            return self.rank_applicants(requirements, applicants[:top_k])
        
        # Filter out organization documents - only keep applicant documents
        applicant_sources = {applicant.metadata.get('source') for applicant in applicants if applicant.metadata and applicant.metadata.get('source')}
        filtered_docs = [doc for doc in similar_docs if doc.metadata and doc.metadata.get('source') in applicant_sources]
        
        # If no filtered docs, use all applicants as fallback
        if not filtered_docs:
            print("No filtered docs found, using all applicants")
            filtered_docs = applicants[:top_k]
        
        # Limit to top_k results
        filtered_docs = filtered_docs[:top_k]
        
        # Calculate similarity scores for each document using cosine similarity
        results = []
        for doc in filtered_docs:
            # Extract embeddings from metadata
            req_embedding = self._get_document_embedding(requirements[0])  # Use first requirement as reference
            applicant_embedding = self._get_document_embedding(doc)
            
            if req_embedding is not None and applicant_embedding is not None:
                # Calculate cosine similarity
                score = cosine_similarity([req_embedding], [applicant_embedding])[0][0]
            else:
                # Fallback to text-based similarity
                score = self._calculate_similarity(combined_query, doc.page_content)
            
            # Add similarity score to document metadata
            doc_with_score = Document(
                page_content=doc.page_content,
                metadata=doc.metadata.copy() if doc.metadata else {}
            )
            doc_with_score.metadata['similarity_score'] = float(score)
            
            results.append((doc_with_score, score))
        
        # If we don't have enough results, add remaining applicants with low scores
        if len(results) < top_k:
            existing_sources = {doc.metadata.get('source') for doc, _ in results if doc.metadata and doc.metadata.get('source')}
            remaining_applicants = [applicant for applicant in applicants 
                                  if applicant.metadata and applicant.metadata.get('source') not in existing_sources]
            
            for applicant in remaining_applicants[:top_k - len(results)]:
                # Calculate similarity score
                req_embedding = self._get_document_embedding(requirements[0])
                applicant_embedding = self._get_document_embedding(applicant)
                
                if req_embedding is not None and applicant_embedding is not None:
                    score = cosine_similarity([req_embedding], [applicant_embedding])[0][0]
                else:
                    score = self._calculate_similarity(combined_query, applicant.page_content)
                
                # Add similarity score to document metadata
                applicant_with_score = Document(
                    page_content=applicant.page_content,
                    metadata=applicant.metadata.copy() if applicant.metadata else {}
                )
                applicant_with_score.metadata['similarity_score'] = float(score)
                
                results.append((applicant_with_score, score))
        
        # Sort by similarity score (descending) and limit to top_k
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:top_k]
        
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
        # For each applicant, calculate similarity to requirements
        ranked_applicants = []
        
        # Combine all requirement texts into a single query
        requirement_texts = [req.page_content for req in requirements]
        combined_query = " ".join(requirement_texts)
        
        # Get requirement embedding for comparison
        req_embedding = self._get_document_embedding(requirements[0]) if requirements else None
        
        # Search for each applicant document
        for applicant in applicants:
            # Extract applicant embedding
            applicant_embedding = self._get_document_embedding(applicant)
            
            if req_embedding is not None and applicant_embedding is not None:
                # Calculate cosine similarity using embeddings
                score = cosine_similarity([req_embedding], [applicant_embedding])[0][0]
            else:
                # Fallback to text-based similarity
                score = self._calculate_similarity(combined_query, applicant.page_content)
            
            # Add similarity score to document metadata
            applicant_with_score = Document(
                page_content=applicant.page_content,
                metadata=applicant.metadata.copy() if applicant.metadata else {}
            )
            applicant_with_score.metadata['similarity_score'] = float(score)
            
            ranked_applicants.append((applicant_with_score, score))
        
        # Sort by similarity score (descending)
        ranked_applicants.sort(key=lambda x: x[1], reverse=True)
        
        return ranked_applicants
    
    def _get_document_embedding(self, document: Document) -> List[float]:
        """
        Extract embedding from document metadata
        
        Args:
            document: Document object
            
        Returns:
            Embedding vector or None if not found
        """
        if hasattr(document, 'metadata') and document.metadata:
            return document.metadata.get('embedding')
        return None
    
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