"""
Argo API Embeddings Client - Clean adapter for text embeddings.

This module provides a LangChain-compatible embeddings model that uses the Argo API
for generating text embeddings. Supports batching for efficient processing.
"""
import httpx
import logging
from typing import List, Optional
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class ArgoEmbeddingsClient(Embeddings):
    """
    Argo API Embeddings wrapper for LangChain.
    
    Provides clean integration with Argo API for generating text embeddings,
    with automatic batching and robust error handling.
    
    Attributes:
        base_url: Base URL for Argo API
        model: Embeddings model name (e.g., 'v3large', 'v3small')
        username: Argo API username
        batch_size: Number of texts to process per API call
        timeout: Request timeout in seconds
    
    Example:
        ```python
        from src.integrations.argo import ArgoEmbeddingsClient
        
        embeddings = ArgoEmbeddingsClient(
            model="v3large",
            username="your-username",
            batch_size=16
        )
        
        # Embed documents
        doc_embeddings = embeddings.embed_documents([
            "The the facility is operating normally",
            "Beam current is 102 mA"
        ])
        
        # Embed query
        query_embedding = embeddings.embed_query("What is the beam status?")
        ```
    """
    
    base_url: str = "https://your-argo-gateway.example.com/api/v1/resource"
    model: str = "v3large"
    username: str = "your-username"
    batch_size: int = 16
    timeout: float = 240.0
    
    def __init__(
        self,
        base_url: str = "https://your-argo-gateway.example.com/api/v1/resource",
        model: str = "v3large",
        username: str = "your-username",
        batch_size: int = 16,
        timeout: float = 240.0
    ):
        """
        Initialize Argo Embeddings Client.
        
        Args:
            base_url: Base URL for Argo API
            model: Embeddings model name
            username: Argo API username
            batch_size: Number of texts per batch
            timeout: Request timeout in seconds
        """
        self.base_url = base_url
        self.model = model
        self.username = username
        self.batch_size = batch_size
        self.timeout = timeout
        self.embed_url = f"{base_url}/embed/"
        
        logger.info(
            f"Initialized ArgoEmbeddingsClient: model={model}, "
            f"batch_size={batch_size}, timeout={timeout}s"
        )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed search docs using Argo API with batching.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (one per input text)
            
        Raises:
            Exception: If API call fails
        """
        if not texts:
            logger.warning("Empty text list provided to embed_documents")
            return []
        
        logger.info(f"Embedding {len(texts)} documents in batches of {self.batch_size}")
        all_embeddings = []
        
        for batch_idx in range(0, len(texts), self.batch_size):
            batch_texts = texts[batch_idx:batch_idx + self.batch_size]
            batch_num = (batch_idx // self.batch_size) + 1
            total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
            
            logger.debug(
                f"Processing batch {batch_num}/{total_batches} "
                f"({len(batch_texts)} texts)"
            )
            
            payload = {
                "user": self.username,
                "model": self.model,
                "prompt": batch_texts
            }
            
            try:
                response = httpx.post(
                    self.embed_url, 
                    json=payload, 
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                result = response.json()
                batch_embeddings = result.get("embedding", [])
                
                if len(batch_embeddings) != len(batch_texts):
                    logger.warning(
                        f"Expected {len(batch_texts)} embeddings, "
                        f"got {len(batch_embeddings)}"
                    )
                
                all_embeddings.extend(batch_embeddings)
                logger.debug(f"Batch {batch_num} completed successfully")
                
            except httpx.TimeoutException as e:
                logger.error(f"Argo embeddings API timeout: {e}")
                raise Exception(f"Argo embeddings timeout: {e}")
            except httpx.HTTPStatusError as e:
                logger.error(f"Argo embeddings API HTTP error: {e}")
                raise Exception(f"Argo embeddings HTTP error: {e}")
            except Exception as e:
                logger.error(f"Argo embeddings API error: {e}")
                raise e
        
        logger.info(f"✅ Successfully embedded {len(all_embeddings)} documents")
        return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed query text using Argo API.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector for the query
            
        Raises:
            Exception: If API call fails
        """
        if not text:
            logger.warning("Empty text provided to embed_query")
            return []
        
        logger.debug(f"Embedding query: {text[:100]}...")
        embeddings = self.embed_documents([text])
        return embeddings[0] if embeddings else []
