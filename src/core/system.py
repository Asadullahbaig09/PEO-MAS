import json
from datetime import datetime
from typing import List
import logging

from src.ingestion.ingestion_layer import IngestionLayer
from src.knowledge.graph import TimeAwareKnowledgeGraph
from src.knowledge.embeddings import EmbeddingEngine
from src.cognitive.anomaly_detector import AnomalyDetector, EthicalAgent
from src.meta.generator import MetaAgentGenerator
from src.meta.registry import AgentRegistry
from src.models.agent import AgentSpecification
from src.models.anomaly import AnomalyReport

# RAG imports
from src.rag.vector_store import VectorStore
from src.rag.retriever import DocumentRetriever
from src.rag.generator import EthicalAssessmentGenerator
from src.rag.document_processor import DocumentProcessor, create_sample_ethical_policies

# Legal compliance imports
from src.meta.law_checker import LawChecker
from src.meta.law_generator import LawGenerator

from config.settings import settings

logger = logging.getLogger(__name__)


class PerpetualEthicalOversightMAS:
    """Main system orchestrating all components with Multi-Agent RAG"""
    
    def __init__(self, enable_rag: bool = True):
        print("="*70)
        print("Initializing Perpetual Ethical Oversight MAS with RAG Support")
        print("="*70)
        
        # Core layers
        self.ingestion = IngestionLayer()
        self.knowledge_graph = TimeAwareKnowledgeGraph()
        # Anomaly detector will be initialized after RAG (needs law checker/generator)
        self.anomaly_detector = None
        
        # Shared LLM Interface (single instance to save VRAM)
        self.llm_interface = None
        
        # RAG Components
        self.rag_enabled = enable_rag
        self.vector_store = None
        self.retriever = None
        self.rag_generator = None
        self.doc_processor = None
        
        # Legal compliance components
        self.law_checker = None
        self.law_generator = None
        
        # Initialize shared LLM (only once for entire system)
        if self.rag_enabled:
            self._initialize_shared_llm()
        
        # Meta generator (uses shared LLM)
        self.meta_generator = MetaAgentGenerator(self.knowledge_graph, llm_interface=self.llm_interface)
        
        if self.rag_enabled:
            self._initialize_rag_system()
            self._initialize_legal_system()
        
        # Anomaly detector (with legal components)
        self.anomaly_detector = AnomalyDetector(
            law_checker=self.law_checker,
            law_generator=self.law_generator
        )
        
        # Agent Registry (with RAG support)
        self.agent_registry = AgentRegistry(
            retriever=self.retriever,
            generator=self.rag_generator
        )
        
        # Initialize with base agents
        self._initialize_base_agents()
        
        # Metrics
        self.metrics = {
            'signals_processed': 0,
            'anomalies_detected': 0,
            'system_evolution_events': [],
            'rag_assessments': 0,
            'total_documents_retrieved': 0,
            'legal_recommendations_generated': 0,
            'laws_checked': 0
        }
        
        print("="*70)
        print("✓ System initialization complete")
        print(f"  RAG Enabled: {self.rag_enabled}")
        print(f"  Legal Compliance System: {'Enabled' if self.law_checker else 'Disabled'}")
        if self.rag_enabled and self.vector_store:
            domains = self.vector_store.list_all_domains()
            print(f"  Knowledge Domains: {len(domains)}")
        print("="*70 + "\n")
    
    def _initialize_rag_system(self):
        """Initialize RAG components and load initial knowledge base"""
        try:
            logger.info("Initializing RAG system...")
            
            # Create vector store
            self.vector_store = VectorStore(persist_directory="./data/chromadb")
            
            # Create embedding engine
            embedding_engine = EmbeddingEngine()
            
            # Create retriever
            self.retriever = DocumentRetriever(
                vector_store=self.vector_store,
                default_k=5,
                use_reranking=True
            )
            
            # Create assessment generator (uses shared LLM)
            self.rag_generator = EthicalAssessmentGenerator(llm_interface=self.llm_interface)
            
            # Create document processor
            self.doc_processor = DocumentProcessor(
                vector_store=self.vector_store,
                embedding_engine=embedding_engine,
                chunk_size=500,
                chunk_overlap=50
            )
            
            # Ingest sample ethical policies
            logger.info("Loading ethical policy knowledge base...")
            policies = create_sample_ethical_policies()
            logger.info(f"Created {len(policies)} sample policies, starting ingestion...")
            stats = self.doc_processor.ingest_policy_documents(policies)
            
            logger.info(
                f"✓ RAG system initialized: {stats['successful']}/{stats['total']} "
                f"policies loaded ({stats.get('failed', 0)} failed, {stats.get('duplicates', 0)} duplicates)"
            )
            logger.info("RAG initialization complete, continuing with system setup...")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            logger.warning("Continuing without RAG support")
            self.rag_enabled = False
            self.vector_store = None
            self.retriever = None
            self.rag_generator = None
    
    def _initialize_shared_llm(self):
        """Initialize single shared LLM instance for entire system"""
        try:
            logger.info("Initializing shared LLM interface...")
            from src.meta.llm_interface import LLMInterface
            self.llm_interface = LLMInterface()
            if self.llm_interface.available:
                logger.info("✓ Shared fine-tuned LLM loaded successfully")
            else:
                logger.warning("⚠ LLM unavailable - components will use template fallback")
                self.llm_interface = None
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            self.llm_interface = None
    
    def _initialize_legal_system(self):
        """Initialize legal compliance checking and generation components"""
        try:
            logger.info("Initializing legal compliance system...")
            
            # Create law checker (needs retriever for RAG)
            self.law_checker = LawChecker(retriever=self.retriever)
            
            # Create law generator (uses shared LLM instance)
            self.law_generator = LawGenerator(llm_interface=self.llm_interface)
            
            if self.llm_interface and self.llm_interface.available:
                logger.info("✓ Legal system initialized with LLM support")
            else:
                logger.info("✓ Legal system initialized with template fallback")
                
        except Exception as e:
            logger.error(f"Failed to initialize legal system: {e}")
            logger.warning("Continuing without legal compliance checking")
            self.law_checker = None
            self.law_generator = None
        
    def _initialize_base_agents(self):
        """Create initial agent pool — one agent per ethical domain"""
        base_specs = [
            AgentSpecification(
                agent_id="agent_0_fairness",
                name="Fairness Monitor",
                domain="bias",
                capabilities=["bias", "fairness", "discrimination"],
                prompt_template="Monitor for fairness and bias issues",
                success_metrics={'coverage_target': 0.8},
                tools=['search', 'analyze', 'statistical_analysis'],
                created_at=datetime.now()
            ),
            AgentSpecification(
                agent_id="agent_0_privacy",
                name="Privacy Guardian",
                domain="privacy",
                capabilities=["privacy", "data_protection"],
                prompt_template="Monitor privacy and data protection",
                success_metrics={'coverage_target': 0.8},
                tools=['search', 'analyze', 'data_flow_analysis'],
                created_at=datetime.now()
            ),
            AgentSpecification(
                agent_id="agent_0_general",
                name="General AI Ethics Overseer",
                domain="general",
                capabilities=["general", "governance", "policy"],
                prompt_template="Monitor general AI governance and ethics policy",
                success_metrics={'coverage_target': 0.8},
                tools=['search', 'analyze', 'policy_analysis'],
                created_at=datetime.now()
            ),
            AgentSpecification(
                agent_id="agent_0_security",
                name="Security Compliance Monitor",
                domain="security",
                capabilities=["security", "cybersecurity", "vulnerability", "breach"],
                prompt_template="Monitor AI system security and cyber compliance",
                success_metrics={'coverage_target': 0.8},
                tools=['search', 'analyze', 'threat_analysis'],
                created_at=datetime.now()
            ),
            AgentSpecification(
                agent_id="agent_0_transparency",
                name="Transparency Monitor",
                domain="transparency",
                capabilities=["transparency", "explainability", "interpretability"],
                prompt_template="Monitor AI transparency and explainability requirements",
                success_metrics={'coverage_target': 0.8},
                tools=['search', 'analyze', 'audit'],
                created_at=datetime.now()
            ),
            AgentSpecification(
                agent_id="agent_0_safety",
                name="Safety Inspector",
                domain="safety",
                capabilities=["safety", "harm", "risk"],
                prompt_template="Monitor AI safety, harm prevention and risk management",
                success_metrics={'coverage_target': 0.8},
                tools=['search', 'analyze', 'risk_assessment'],
                created_at=datetime.now()
            ),
            AgentSpecification(
                agent_id="agent_0_accountability",
                name="Accountability Reviewer",
                domain="accountability",
                capabilities=["accountability", "liability", "responsibility", "audit"],
                prompt_template="Monitor AI accountability, liability and audit trails",
                success_metrics={'coverage_target': 0.8},
                tools=['search', 'analyze', 'audit'],
                created_at=datetime.now()
            ),
        ]
        
        for spec in base_specs:
            self.agent_registry.register(spec)
            self.knowledge_graph.add_node(spec.domain, initial_state=0.5)
    
    def process_cycle(self):
        """Single processing cycle - the heart of the system"""
        print("\n" + "="*70)
        print(f"PROCESSING CYCLE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        # 1. Ingest signals
        signals = self.ingestion.collect_signals()
        print(f"\n[INGESTION] Collected {len(signals)} signals")
        
        for signal in signals:
            self.metrics['signals_processed'] += 1
            
            print(f"\n[SIGNAL] {signal.content[:60]}...")
            print(f"  Category: {signal.category} | Severity: {signal.severity:.2f}")
            
            # 2. Update knowledge graph
            self.knowledge_graph.update_node(signal.category, signal.severity)
            
            # 3. Check for anomalies
            agents = self.agent_registry.get_all_agents()
            anomaly = self.anomaly_detector.detect_anomaly(signal, agents)
            
            if anomaly:
                self.metrics['anomalies_detected'] += 1
                print(f"\n⚠️  [ANOMALY DETECTED]")
                print(f"  Explanation Score: {anomaly.explanation_score:.2f}")
                print(f"  Unexplained: {anomaly.unexplained_factors}")
                
                # 4. Check if legal recommendation was generated
                legal_rec = anomaly.context.get('legal_recommendation')
                if legal_rec:
                    self.metrics['legal_recommendations_generated'] += 1
                    print(f"\n⚖️  [LEGAL RECOMMENDATION GENERATED]")
                    print(f"  Title: {legal_rec['title']}")
                    print(f"  Severity: {legal_rec['severity']:.2f}")
                    print(f"  Proposed Law Preview: {legal_rec['proposed_law'][:150]}...")
                else:
                    print(f"  ✓ Existing laws provide adequate coverage")
                
                # Track for metrics
                if anomaly.context.get('legal_check_performed'):
                    self.metrics['laws_checked'] += 1
        
        # Print metrics
        self._update_rag_metrics()
        self._print_metrics()
    
    def check_for_system_evolution(
        self, 
        anomaly: AnomalyReport, 
        current_agent_pool: List[EthicalAgent]
    ) -> bool:
        """
        Core evolution logic - spawns new agents when needed
        This is the key innovation of the system!
        """
        explanation_score = anomaly.explanation_score
        
        if explanation_score < settings.ETHICAL_COVERAGE_THRESHOLD:
            print("\n🌟 [EVOLUTION] New Ethical Frontier Detected")
            print("   Spawning specialized auditor...")
            
            # Structural synthesis
            new_agent_spec = self.meta_generator.synthesize(anomaly)
            
            # Deployment into MAS
            new_agent = self.agent_registry.register(new_agent_spec)
            
            # Update knowledge graph
            self.knowledge_graph.add_node(new_agent_spec.domain, initial_state=0.6)
            
            # Log evolution event
            self.metrics['system_evolution_events'].append({
                'timestamp': datetime.now().isoformat(),
                'trigger': anomaly.anomaly_id,
                'new_agent': new_agent_spec.agent_id,
                'reason': anomaly.unexplained_factors
            })
            
            print(f"   ✓ Agent '{new_agent_spec.name}' deployed successfully")
            return True
        
        return False
    
    def _update_rag_metrics(self):
        """Collect RAG metrics from all agents"""
        if not self.rag_enabled:
            return
        
        total_rag_assessments = 0
        total_docs_retrieved = 0
        
        for agent in self.agent_registry.get_all_agents():
            if hasattr(agent, 'rag_assessments'):
                total_rag_assessments += agent.rag_assessments
            if hasattr(agent, 'total_docs_retrieved'):
                total_docs_retrieved += agent.total_docs_retrieved
        
        self.metrics['rag_assessments'] = total_rag_assessments
        self.metrics['total_documents_retrieved'] = total_docs_retrieved
    
    def _print_metrics(self):
        """Print current metrics"""
        print(f"\n[METRICS]")
        print(f"  Total Signals: {self.metrics['signals_processed']}")
        print(f"  Anomalies: {self.metrics['anomalies_detected']}")
        
        if self.rag_enabled:
            print(f"  RAG Assessments: {self.metrics.get('rag_assessments', 0)}")
            print(f"  Documents Retrieved: {self.metrics.get('total_documents_retrieved', 0)}")
        
        # Legal compliance metrics
        if self.law_checker:
            print(f"  Laws Checked: {self.metrics.get('laws_checked', 0)}")
            print(f"  Legal Recommendations: {self.metrics.get('legal_recommendations_generated', 0)}")
    
    def get_rag_statistics(self) -> dict:
        """Get RAG system statistics"""
        if not self.rag_enabled:
            return {'rag_enabled': False}
        
        stats = {
            'rag_enabled': True,
            'retriever_stats': self.retriever.get_statistics() if self.retriever else {},
            'processor_stats': self.doc_processor.get_statistics() if self.doc_processor else {},
            'rag_assessments': self.metrics.get('rag_assessments', 0),
            'total_documents_retrieved': self.metrics.get('total_documents_retrieved', 0)
        }
        
        return stats
    
    def export_system_state(self) -> dict:
        """Export complete system state"""
        return {
            'metrics': self.metrics,
            'knowledge_graph': self.knowledge_graph.export_state(),
            'agents': [
                agent.spec.to_dict() 
                for agent in self.agent_registry.get_all_agents()
            ],
            'timestamp': datetime.now().isoformat()
        }