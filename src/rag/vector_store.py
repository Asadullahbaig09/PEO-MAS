"""
Vector Store for Multi-Agent RAG System

Uses ChromaDB for local vector storage (100% free!)
Each ethical domain has its own collection for efficient retrieval.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import numpy as np

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.api.types import EmbeddingFunction
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logging.warning("ChromaDB not installed. Install with: pip install chromadb")

try:
    from sentence_transformers import SentenceTransformer
    import torch
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logging.warning("sentence-transformers not available")

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Vector database for storing and retrieving ethical documents
    
    Features:
    - Domain-specific collections (privacy, bias, transparency, etc.)
    - Persistent storage
    - Metadata filtering
    - Hybrid search (semantic + keyword)
    """
    
    def __init__(self, persist_directory: str = "./data/chromadb"):
        """
        Initialize vector store
        
        Args:
            persist_directory: Directory for persistent storage
        """
        from config.settings import settings
        
        self.persist_directory = persist_directory
        self.client = None
        self.collections = {}
        self.embedding_model = None
        self.embedding_function = None
        
        # Initialize local embedding model from configured path
        if EMBEDDINGS_AVAILABLE:
            try:
                model_path = Path(settings.EMBEDDING_MODEL_PATH)
                
                # Detect GPU availability
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                if device == 'cuda':
                    logger.info(f"✓ GPU detected - using CUDA for embeddings")
                else:
                    logger.info("Using CPU for embeddings (install CUDA for GPU acceleration)")
                
                # Check if model exists locally
                if model_path.exists():
                    logger.info(f"Loading embedding model from: {model_path}")
                    self.embedding_model = SentenceTransformer(str(model_path), device=device)
                    logger.info("✓ Local embedding model loaded successfully")
                else:
                    # Download model to specified path if it doesn't exist
                    logger.info(f"Downloading embedding model '{settings.EMBEDDING_MODEL}' to: {model_path}")
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL, cache_folder=str(model_path.parent), device=device)
                    logger.info("✓ Embedding model downloaded and cached")
                    
                # Create custom ChromaDB embedding function
                self.embedding_function = self._create_embedding_function()
                    
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
                logger.warning("Embedding model will be downloaded on first use")
                self.embedding_model = None
        
        if CHROMADB_AVAILABLE:
            try:
                # Use PersistentClient for data persistence
                self.client = chromadb.PersistentClient(path=persist_directory)
                logger.info(f"✓ ChromaDB initialized at {persist_directory}")
            except Exception as e:
                logger.error(f"Failed to initialize ChromaDB: {e}")
                self.client = None
        else:
            logger.warning("ChromaDB not available - using in-memory fallback")
            self._init_fallback_storage()
    
    def _create_embedding_function(self):
        """Create custom ChromaDB embedding function using local model"""
        if not self.embedding_model:
            return None
        
        # Use try-except to handle both old and new ChromaDB versions
        try:
            # Try new ChromaDB API with base class
            from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
            
            model = self.embedding_model
            
            class LocalEmbeddingFunction(EmbeddingFunction):
                def __call__(self, input: Documents) -> Embeddings:
                    """Generate embeddings for input texts"""
                    embeddings = model.encode(input, convert_to_numpy=True)
                    return embeddings.tolist()
            
            return LocalEmbeddingFunction()
            
        except ImportError:
            # Fallback for older ChromaDB versions
            model = self.embedding_model
            
            class LocalEmbeddingFunction:
                def __call__(self, input: list[str]) -> list[list[float]]:
                    """Generate embeddings for input texts"""
                    embeddings = model.encode(input, convert_to_numpy=True)
                    return embeddings.tolist()
            
            return LocalEmbeddingFunction()
    
    def _init_fallback_storage(self):
        """Initialize in-memory storage if ChromaDB unavailable"""
        self.fallback_storage = {}  # domain -> List[{id, text, embedding, metadata}]
    
    def get_or_create_collection(self, domain: str) -> Any:
        """
        Get or create collection for ethical domain
        
        Args:
            domain: Ethical domain (privacy, bias, transparency, etc.)
        
        Returns:
            ChromaDB collection or fallback storage
        """
        if domain in self.collections:
            return self.collections[domain]
        
        if self.client:
            try:
                # Use custom embedding function if available, otherwise None for pre-computed embeddings
                collection = self.client.get_or_create_collection(
                    name=f"ethical_{domain}",
                    metadata={"domain": domain},
                    embedding_function=self.embedding_function  # Uses local model, no downloads
                )
                self.collections[domain] = collection
                # logger.info(f"✓ Collection created for domain: {domain}")
                return collection
            except Exception as e:
                logger.error(f"Error creating collection for {domain}: {e}")
                return None
        else:
            # Fallback
            if domain not in self.fallback_storage:
                self.fallback_storage[domain] = []
            return domain
    
    def add_documents(
        self,
        domain: str,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: Optional[List[np.ndarray]] = None,
        ids: Optional[List[str]] = None
    ) -> bool:
        """
        Add documents to domain collection
        
        Args:
            domain: Ethical domain
            documents: List of document texts
            metadatas: List of metadata dicts
            embeddings: Optional pre-computed embeddings
            ids: Optional document IDs
        
        Returns:
            Success status
        """
        collection = self.get_or_create_collection(domain)
        
        if not collection:
            return False
        
        # Generate IDs if not provided
        if ids is None:
            ids = [f"{domain}_{i}_{datetime.now().timestamp()}" 
                   for i in range(len(documents))]
        
        try:
            if self.client:
                # Use ChromaDB
                if embeddings is not None:
                    collection.add(
                        documents=documents,
                        metadatas=metadatas,
                        ids=ids,
                        embeddings=[emb.tolist() for emb in embeddings]
                    )
                else:
                    collection.add(
                        documents=documents,
                        metadatas=metadatas,
                        ids=ids
                    )
                # logger.info(f"✓ Added {len(documents)} documents to {domain} collection")
            else:
                # Fallback storage
                for i, doc in enumerate(documents):
                    self.fallback_storage[domain].append({
                        'id': ids[i],
                        'text': doc,
                        'metadata': metadatas[i],
                        'embedding': embeddings[i] if embeddings else None
                    })
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to {domain}: {e}")
            return False
    
    def search(
        self,
        domain: str,
        query_text: str,
        query_embedding: Optional[np.ndarray] = None,
        n_results: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Search for relevant documents in domain collection
        
        Args:
            domain: Ethical domain to search
            query_text: Query text
            query_embedding: Optional pre-computed query embedding
            n_results: Number of results to return
            metadata_filter: Optional metadata filters
        
        Returns:
            Search results with documents, distances, metadatas
        """
        collection = self.get_or_create_collection(domain)
        
        if not collection:
            return {'documents': [], 'metadatas': [], 'distances': []}
        
        try:
            if self.client:
                # Use ChromaDB
                if query_embedding is not None:
                    results = collection.query(
                        query_embeddings=[query_embedding.tolist()],
                        n_results=n_results,
                        where=metadata_filter
                    )
                else:
                    # Generate embedding for query_text using local model
                    if self.embedding_model:
                        try:
                            query_embedding = self.embedding_model.encode(query_text, convert_to_numpy=True)
                            results = collection.query(
                                query_embeddings=[query_embedding.tolist()],
                                n_results=n_results,
                                where=metadata_filter
                            )
                        except Exception as e:
                            logger.error(f"Failed to generate embedding: {e}")
                            return {'documents': [], 'metadatas': [], 'distances': []}
                    else:
                        logger.warning("No embedding model available, cannot search by text")
                        return {'documents': [], 'metadatas': [], 'distances': []}
                
                return {
                    'documents': results['documents'][0] if results['documents'] else [],
                    'metadatas': results['metadatas'][0] if results['metadatas'] else [],
                    'distances': results['distances'][0] if results['distances'] else []
                }
            else:
                # Fallback: simple keyword search
                docs = self.fallback_storage.get(domain, [])
                query_lower = query_text.lower()
                
                scored_docs = []
                for doc in docs:
                    # Simple keyword matching score
                    text_lower = doc['text'].lower()
                    score = sum(1 for word in query_lower.split() if word in text_lower)
                    scored_docs.append((score, doc))
                
                # Sort by score and take top n_results
                scored_docs.sort(reverse=True, key=lambda x: x[0])
                top_docs = scored_docs[:n_results]
                
                return {
                    'documents': [doc['text'] for _, doc in top_docs],
                    'metadatas': [doc['metadata'] for _, doc in top_docs],
                    'distances': [1.0 / (score + 1) for score, _ in top_docs]
                }
                
        except Exception as e:
            logger.error(f"Error searching {domain}: {e}")
            return {'documents': [], 'metadatas': [], 'distances': []}
    
    def get_collection_stats(self, domain: str) -> Dict[str, Any]:
        """Get statistics for domain collection"""
        collection = self.get_or_create_collection(domain)
        
        if not collection:
            return {'count': 0, 'domain': domain}
        
        try:
            if self.client:
                count = collection.count()
                return {
                    'domain': domain,
                    'count': count,
                    'name': collection.name
                }
            else:
                docs = self.fallback_storage.get(domain, [])
                return {
                    'domain': domain,
                    'count': len(docs)
                }
        except Exception as e:
            logger.error(f"Error getting stats for {domain}: {e}")
            return {'count': 0, 'domain': domain}
    
    def list_all_domains(self) -> List[str]:
        """List all available domain collections"""
        if self.client:
            try:
                collections = self.client.list_collections()
                return [c.name.replace('ethical_', '') for c in collections]
            except:
                return list(self.collections.keys())
        else:
            return list(self.fallback_storage.keys())
    
    def delete_collection(self, domain: str) -> bool:
        """Delete domain collection"""
        try:
            if self.client:
                self.client.delete_collection(f"ethical_{domain}")
                if domain in self.collections:
                    del self.collections[domain]
            else:
                if domain in self.fallback_storage:
                    del self.fallback_storage[domain]
            
            logger.info(f"✓ Deleted collection for domain: {domain}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection {domain}: {e}")
            return False
