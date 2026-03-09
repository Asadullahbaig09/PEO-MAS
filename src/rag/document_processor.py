"""
Document Processor for Multi-Agent RAG System

Ingests and processes ethical documents into vector store
Supports multiple document formats and sources
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import hashlib

from src.rag.vector_store import VectorStore
from src.knowledge.embeddings import EmbeddingEngine

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Processes and ingests ethical documents into vector store
    
    Features:
    - Document chunking with context preservation
    - Metadata extraction
    - Embedding generation
    - Batch processing
    - Deduplication
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_engine: Optional[EmbeddingEngine] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        Initialize document processor
        
        Args:
            vector_store: Vector store instance
            embedding_engine: Optional embedding engine
            chunk_size: Characters per chunk
            chunk_overlap: Overlap between chunks
        """
        self.vector_store = vector_store
        self.embedding_engine = embedding_engine or EmbeddingEngine()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Track processed documents
        self.processed_hashes = set()
        
        logger.info("✓ Document processor initialized")
    
    def ingest_document(
        self,
        content: str,
        domain: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Ingest single document
        
        Args:
            content: Document text
            domain: Ethical domain (privacy, bias, etc.)
            metadata: Document metadata
        
        Returns:
            Success status
        """
        try:
            # Check for duplicates
            doc_hash = self._hash_content(content)
            if doc_hash in self.processed_hashes:
                logger.debug(f"Skipping duplicate document: {metadata.get('title', 'Untitled')}")
                return False
            
            # logger.info(f"  → Chunking document...")
            # Chunk document
            chunks = self._chunk_document(content)
            
            # logger.info(f"  → Generating embeddings for {len(chunks)} chunks...")
            # Generate embeddings
            embeddings = self.embedding_engine.encode(chunks)
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            
            # logger.info(f"  → Preparing metadata...")
            # Add metadata to each chunk
            chunk_metadatas = []
            for i, chunk in enumerate(chunks):
                chunk_meta = metadata.copy()
                chunk_meta['chunk_id'] = i
                chunk_meta['total_chunks'] = len(chunks)
                chunk_meta['ingested_at'] = datetime.now().isoformat()
                chunk_metadatas.append(chunk_meta)
            
            # Generate IDs
            doc_id_base = metadata.get('id', doc_hash[:16])
            chunk_ids = [f"{doc_id_base}_chunk_{i}" for i in range(len(chunks))]
            
            # logger.info(f"  → Adding to vector store...")
            # Add to vector store
            success = self.vector_store.add_documents(
                domain=domain,
                documents=chunks,
                metadatas=chunk_metadatas,
                embeddings=embeddings,
                ids=chunk_ids
            )
            
            if success:
                self.processed_hashes.add(doc_hash)
                logger.info(
                    f"✓ Ingested: {metadata.get('title', 'Untitled')} "
                    f"({len(chunks)} chunks) → {domain}"
                )
            
            return success
            
        except Exception as e:
            logger.error(f"Error in ingest_document: {e}", exc_info=True)
            return False
    
    def ingest_documents_batch(
        self,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """
        Ingest multiple documents
        
        Args:
            documents: List of dicts with 'content', 'domain', 'metadata'
        
        Returns:
            Statistics dict
        """
        stats = {
            'total': len(documents),
            'successful': 0,
            'failed': 0,
            'duplicates': 0
        }
        
        for i, doc in enumerate(documents):
            try:
                title = doc.get('metadata', {}).get('title', 'Untitled')
                # logger.info(f"Processing document {i+1}/{len(documents)}: {title}")
                content = doc['content']
                domain = doc['domain']
                metadata = doc.get('metadata', {})
                
                # Check if duplicate
                doc_hash = self._hash_content(content)
                if doc_hash in self.processed_hashes:
                    stats['duplicates'] += 1
                    logger.info(f"Skipped duplicate document {i+1}")
                    continue
                
                success = self.ingest_document(content, domain, metadata)
                if success:
                    stats['successful'] += 1
                else:
                    stats['failed'] += 1
                    logger.warning(f"Failed to ingest document {i+1}: {title}")
                    
            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.error(f"Error ingesting document {i+1}: {e}", exc_info=True)
                stats['failed'] += 1
        
        logger.info(
            f"Batch ingestion complete: {stats['successful']}/{stats['total']} successful, "
            f"{stats['duplicates']} duplicates, {stats['failed']} failed"
        )
        
        return stats
    
    def ingest_from_directory(
        self,
        directory: str,
        domain: str,
        file_pattern: str = "*.txt",
        metadata_extractor: Optional[callable] = None
    ) -> Dict[str, int]:
        """
        Ingest all documents from directory
        
        Args:
            directory: Directory path
            domain: Ethical domain
            file_pattern: Glob pattern for files
            metadata_extractor: Optional function to extract metadata from filename
        
        Returns:
            Statistics dict
        """
        dir_path = Path(directory)
        
        if not dir_path.exists():
            logger.error(f"Directory not found: {directory}")
            return {'total': 0, 'successful': 0, 'failed': 0}
        
        # Find files
        files = list(dir_path.glob(file_pattern))
        logger.info(f"Found {len(files)} files matching {file_pattern} in {directory}")
        
        # Process each file
        documents = []
        for file_path in files:
            try:
                content = file_path.read_text(encoding='utf-8')
                
                # Extract or generate metadata
                if metadata_extractor:
                    metadata = metadata_extractor(file_path)
                else:
                    metadata = {
                        'title': file_path.stem,
                        'source': 'local_file',
                        'filename': file_path.name
                    }
                
                documents.append({
                    'content': content,
                    'domain': domain,
                    'metadata': metadata
                })
                
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
        
        # Batch ingest
        return self.ingest_documents_batch(documents)
    
    def ingest_policy_documents(self, policies: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Ingest predefined policy documents
        
        Args:
            policies: List of policy dicts with content, domain, metadata
        
        Returns:
            Statistics dict
        """
        return self.ingest_documents_batch(policies)
    
    def _chunk_document(self, content: str) -> List[str]:
        """
        Chunk document with overlap for context preservation
        
        Args:
            content: Document text
        
        Returns:
            List of chunks
        """
        if not content or len(content) == 0:
            return []
        
        # Simple character-based chunking with fixed steps
        chunks = []
        step_size = self.chunk_size - self.chunk_overlap
        
        for i in range(0, len(content), step_size):
            chunk = content[i:i + self.chunk_size].strip()
            if chunk:
                chunks.append(chunk)
        
        return chunks if chunks else [content.strip()]
    
    def _hash_content(self, content: str) -> str:
        """Generate hash for content deduplication"""
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processor statistics"""
        return {
            'processed_documents': len(self.processed_hashes),
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'vector_store_stats': {
                domain: self.vector_store.get_collection_stats(domain)
                for domain in self.vector_store.list_all_domains()
            }
        }


def create_sample_ethical_policies() -> List[Dict[str, Any]]:
    """
    Create reference ethical policy documents derived from established
    international standards and regulatory frameworks (IEEE, NIST, EU,
    OECD) for RAG retrieval during ethical assessments.

    Returns:
        List of policy documents ready for ingestion
    """
    policies = []
    
    # BIAS / FAIRNESS POLICIES
    policies.append({
        'content': """
        AI Fairness and Bias Prevention Policy
        (Derived from IEEE Ethically Aligned Design, EU AI Act Title III, and NIST AI RMF MAP function)
        
        Objective: Ensure AI systems treat all individuals and groups fairly without discrimination,
        in accordance with IEEE Standard 7003 (Algorithmic Bias Considerations) and the EU AI Act
        requirements for high-risk AI systems.
        
        Requirements:
        1. All AI systems must be tested for bias across protected characteristics including race, gender, age, disability, and socioeconomic status (EU AI Act Art. 10).
        2. Training data must be representative of the population the system will serve (EU AI Act Art. 10, NIST AI RMF MP-2).
        3. Fairness metrics (demographic parity, equalized odds, equal opportunity) must be measured and documented (NIST AI RMF MS-2).
        4. Systems showing disparate impact >20% must be remediated before deployment (US EEOC 80% rule, EU AI Act Art. 9).
        5. Regular bias audits must be conducted quarterly for production systems (NIST AI RMF MG-3).
        
        Violations:
        - Deployment of systems with known bias issues
        - Failure to test for fairness before production
        - Inadequate representation in training data
        - Ignoring disparate impact findings
        
        Consequences: Immediate system suspension, mandatory retraining, executive review.
        Penalties under EU AI Act Art. 99: up to EUR 15,000,000 or 3% of annual turnover.
        """,
        'domain': 'bias',
        'metadata': {
            'title': 'AI Fairness and Bias Prevention Policy',
            'source': 'IEEE Ethically Aligned Design / EU AI Act / NIST AI RMF',
            'type': 'policy',
            'version': '2.1',
            'date': '2025-01-15',
            'authority': 'IEEE Standards Association, European Commission, NIST'
        }
    })
    
    # PRIVACY POLICIES
    policies.append({
        'content': """
        Data Privacy and Protection Policy for AI Systems
        (Based on GDPR Articles 5, 6, 25, 35; CCPA Sections 1798.100-1798.199; OECD Privacy Guidelines)
        
        Objective: Protect individual privacy rights and ensure compliance with GDPR, CCPA, and related regulations.
        
        Requirements:
        1. Implement privacy by design in all AI systems from inception (GDPR Art. 25).
        2. Minimize data collection to only what is necessary for the specific purpose (GDPR Art. 5(1)(c) — data minimisation).
        3. Implement differential privacy or other privacy-preserving techniques where feasible (NIST Privacy Framework).
        4. Obtain explicit consent for personal data use in AI training (GDPR Art. 6(1)(a), Art. 9(2)(a)).
        5. Provide clear mechanisms for data access, correction, and deletion (GDPR Arts. 15-17, CCPA Sec. 1798.105-1798.110).
        6. Encrypt all personal data at rest and in transit (GDPR Art. 32).
        7. Conduct Privacy Impact Assessments (PIAs) for all new AI systems (GDPR Art. 35).
        
        Violations:
        - Processing personal data without legal basis
        - Failure to implement privacy-preserving techniques
        - Inadequate data security measures
        - Non-compliance with data subject rights requests
        - Unauthorized data sharing or selling
        
        Consequences: GDPR Art. 83 fines up to EUR 20,000,000 or 4% of annual global turnover.
        CCPA fines: $2,500 per violation, $7,500 per intentional violation.
        """,
        'domain': 'privacy',
        'metadata': {
            'title': 'Data Privacy and Protection Policy',
            'source': 'GDPR (EU 2016/679) / CCPA (Cal. Civ. Code 1798) / OECD Privacy Guidelines',
            'type': 'policy',
            'version': '3.0',
            'date': '2025-06-01',
            'authority': 'European Commission, California Legislature, OECD',
            'regulations': 'GDPR, CCPA, PIPEDA'
        }
    })
    
    # TRANSPARENCY POLICIES
    policies.append({
        'content': """
        AI Transparency and Explainability Policy
        (Based on EU AI Act Title IV Art. 50, GDPR Art. 22, OECD AI Principle 1.3, NIST AI RMF GOVERN)
        
        Objective: Ensure AI systems are transparent, interpretable, and their decisions can be explained to stakeholders.
        
        Requirements:
        1. All high-risk AI systems must provide explanations for their decisions (EU AI Act Art. 13, GDPR Art. 22).
        2. Maintain comprehensive documentation of model architecture, training data, and performance metrics (EU AI Act Art. 11).
        3. Implement interpretability tools (SHAP, LIME, attention visualization) for complex models (NIST AI RMF MS-2).
        4. Provide user-facing explanations in plain language for automated decisions (GDPR Art. 12, Art. 22(3)).
        5. Disclose when users are interacting with AI systems vs. humans (EU AI Act Art. 50).
        6. Make model cards publicly available for customer-facing AI (NIST AI RMF GV-1).
        7. Enable human review of high-stakes automated decisions (EU AI Act Art. 14).
        
        Violations:
        - Black-box decision-making in high-stakes scenarios
        - Inadequate documentation of AI systems
        - Failure to disclose AI usage to users
        - Unable to explain decisions upon request
        
        Consequences: EU AI Act Art. 99 fines up to EUR 15,000,000 or 3% of annual turnover.
        Mandatory system redesign, deployment delay, regulatory scrutiny.
        """,
        'domain': 'transparency',
        'metadata': {
            'title': 'AI Transparency and Explainability Policy',
            'source': 'EU AI Act / GDPR Art. 22 / OECD AI Principles',
            'type': 'policy',
            'version': '1.5',
            'date': '2024-11-20',
            'authority': 'European Commission, OECD'
        }
    })
    
    # ACCOUNTABILITY POLICIES
    policies.append({
        'content': """
        AI Accountability and Responsibility Framework
        (Based on NIST AI RMF GOVERN function, EU AI Act Arts. 9-16, ISO/IEC 42001 AI Management System)
        
        Objective: Establish clear lines of accountability for AI system development, deployment, and outcomes.
        
        Requirements:
        1. Designate an AI System Owner for each production AI system (NIST AI RMF GV-2, EU AI Act Art. 16).
        2. Maintain audit trails of all AI decision-making processes (NIST AI RMF MS-3, NIST CSF DE.AE).
        3. Implement monitoring systems for ongoing performance and ethical compliance (NIST AI RMF MG-3, EU AI Act Art. 9).
        4. Establish incident response procedures for AI failures or harms (NIST CSF RS.MA, NIST AI RMF MG-4).
        5. Conduct regular ethics reviews by independent ethics committee (NIST AI RMF GV-5).
        6. Assign liability for AI-caused harms to appropriate stakeholders (EU AI Act Art. 82, GDPR Art. 82).
        7. Maintain insurance coverage for AI-related risks (EU Product Liability Directive 2024/2853).
        
        Roles and Responsibilities (per NIST AI RMF GV-2):
        - AI System Owner: Overall accountability for system performance and ethics
        - Development Team: Responsible for implementing ethical requirements
        - Ethics Committee: Reviews and approves high-risk systems
        - Compliance Officer: Ensures regulatory adherence
        
        Violations:
        - Unclear accountability structures
        - Inadequate audit trails
        - Failure to respond to identified harms
        - Neglecting ongoing monitoring
        
        Consequences: EU AI Act Art. 99 administrative fines; leadership accountability;
        mandatory governance improvements per ISO/IEC 42001.
        """,
        'domain': 'accountability',
        'metadata': {
            'title': 'AI Accountability and Responsibility Framework',
            'source': 'NIST AI RMF / EU AI Act / ISO/IEC 42001',
            'type': 'policy',
            'version': '2.0',
            'date': '2025-03-10',
            'authority': 'NIST, European Commission, ISO/IEC JTC 1/SC 42'
        }
    })
    
    # SAFETY POLICIES
    policies.append({
        'content': """
        AI Safety and Risk Management Policy
        (Based on EU AI Act Title III Arts. 9-15, NIST AI RMF MAP/MANAGE, ISO/IEC 23894 AI Risk Management)
        
        Objective: Prevent AI systems from causing harm to individuals, society, or the environment.
        
        Requirements:
        1. Conduct comprehensive safety testing before any production deployment (EU AI Act Art. 9, NIST AI RMF MP-4).
        2. Implement fail-safe mechanisms and circuit breakers for critical systems (EU AI Act Art. 14, Art. 15).
        3. Perform adversarial testing and red-teaming exercises (NIST AI RMF MS-2, EU AI Act Art. 15).
        4. Monitor for edge cases and unexpected behaviors in production (NIST AI RMF MG-3, EU AI Act Art. 72).
        5. Maintain human oversight for high-stakes decisions (EU AI Act Art. 14 — human oversight measures).
        6. Implement gradual rollout strategies with monitoring (NIST AI RMF MG-2).
        7. Establish clear criteria for emergency system shutdown (EU AI Act Art. 14(4)(e) — 'stop' button).
        
        Risk Categories (per NIST AI RMF and EU AI Act Annex III):
        - Physical harm: AI controlling physical systems (robots, vehicles, medical devices)
        - Economic harm: Financial decision-making, employment systems
        - Psychological harm: Mental health applications, persuasive systems
        - Social harm: Content moderation, recommendation systems
        
        Violations:
        - Deploying untested or inadequately tested systems
        - Lack of fail-safe mechanisms in critical applications
        - Ignoring identified safety risks
        - Insufficient monitoring of production systems
        
        Consequences: EU AI Act Art. 99 fines up to EUR 35,000,000 or 7% of turnover for
        prohibited practices; immediate system shutdown, comprehensive safety audit.
        """,
        'domain': 'safety',
        'metadata': {
            'title': 'AI Safety and Risk Management Policy',
            'source': 'EU AI Act / NIST AI RMF / ISO/IEC 23894',
            'type': 'policy',
            'version': '1.8',
            'date': '2025-02-01',
            'authority': 'European Commission, NIST, ISO/IEC JTC 1/SC 42'
        }
    })
    
    # GENERAL ETHICS
    policies.append({
        'content': """
        General AI Ethics Principles and Guidelines
        (Synthesised from OECD AI Principles 2019, UNESCO Recommendation on AI Ethics 2021,
        EU Ethics Guidelines for Trustworthy AI 2019, and the Asilomar AI Principles)
        
        Core Principles (per OECD AI Principles and UNESCO Recommendation):
        1. Human-Centered: AI should benefit humanity and respect human rights and dignity (OECD 1.2, UNESCO Art. 12).
        2. Fairness: AI should not discriminate or create unfair advantages/disadvantages (OECD 1.2, UNESCO Art. 39).
        3. Transparency: AI operations should be understandable and explainable (OECD 1.3, UNESCO Art. 18).
        4. Privacy: AI should protect personal information and privacy rights (OECD 1.2, UNESCO Art. 14).
        5. Security: AI should be secure against attacks and misuse (OECD 1.4, EU Cybersecurity Act).
        6. Accountability: Clear responsibility for AI outcomes must exist (OECD 1.5, UNESCO Art. 26).
        7. Reliability: AI should perform consistently and correctly (NIST AI RMF Trustworthiness Characteristic).
        8. Safety: AI should not cause harm to people or the environment (OECD 1.4, EU AI Act Art. 9).
        
        Cross-Cutting Requirements:
        - Ethics impact assessments for all new AI initiatives (UNESCO Art. 16)
        - Stakeholder consultation including affected communities (NIST AI RMF GV-5)
        - Continuous monitoring and improvement (NIST AI RMF MANAGE function)
        - Regular ethics training for all AI practitioners (NIST CSF PR.AT)
        - Whistleblower protections for ethics concerns (EU Directive 2019/1937)
        
        When in doubt, prioritize human welfare and safety over business objectives.
        """,
        'domain': 'general',
        'metadata': {
            'title': 'General AI Ethics Principles',
            'source': 'OECD AI Principles / UNESCO AI Ethics / EU Trustworthy AI Guidelines',
            'type': 'principles',
            'version': '1.0',
            'date': '2024-01-01',
            'authority': 'OECD, UNESCO, European Commission High-Level Expert Group'
        }
    })
    
    return policies
