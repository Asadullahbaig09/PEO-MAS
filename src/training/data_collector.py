"""
Training Data Collection Module

Collects labeled data from system runs for future fine-tuning:
- Signal classifications (anomaly detection)
- Domain predictions (agent assignment)
- Legal recommendation quality ratings
- RAG retrieval relevance scores
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


class TrainingDataCollector:
    """
    Collects and stores training data from system execution
    
    Use Cases:
    1. Fine-tune embedding model for better domain classification
    2. Train anomaly detector (replace threshold-based with ML)
    3. Improve legal recommendation generation
    4. Optimize RAG retrieval quality
    """
    
    def __init__(self, output_dir: str = "./data/training"):
        """
        Initialize data collector
        
        Args:
            output_dir: Directory to save training data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory storage for current session
        self.signal_data = []  # For anomaly detection training
        self.domain_data = []  # For domain classification training
        self.legal_data = []   # For legal recommendation training
        self.rag_data = []     # For RAG retrieval training
        
        logger.info(f"✓ Training data collector initialized: {output_dir}")
    
    def record_signal(
        self,
        signal: Dict[str, Any],
        predicted_domain: str,
        is_anomaly: bool,
        agent_assessment: Optional[Dict[str, Any]] = None,
        human_labels: Optional[Dict[str, Any]] = None
    ):
        """
        Record signal data for training
        
        Args:
            signal: Original signal dict
            predicted_domain: System's predicted domain
            is_anomaly: System's anomaly prediction
            agent_assessment: Agent's RAG-based assessment (if any)
            human_labels: Optional human-provided labels
                {
                    'true_domain': 'privacy',
                    'true_anomaly': True,
                    'severity_rating': 0.85,
                    'notes': 'Confirmed privacy violation'
                }
        """
        record = {
            'timestamp': datetime.now().isoformat(),
            'signal_id': signal.get('id', 'unknown'),
            'title': signal.get('title', '')[:200],  # Truncate for storage
            'content': signal.get('content', '')[:1000],
            'source': signal.get('source', 'unknown'),
            'category': signal.get('category', 'general'),
            'severity': signal.get('severity', 0.5),
            
            # System predictions
            'predicted_domain': predicted_domain,
            'predicted_anomaly': is_anomaly,
            
            # Agent assessment (if available)
            'agent_risk_level': agent_assessment.get('risk_level') if agent_assessment else None,
            'agent_confidence': agent_assessment.get('confidence') if agent_assessment else None,
            
            # Human labels (to be filled later)
            'true_domain': human_labels.get('true_domain') if human_labels else None,
            'true_anomaly': human_labels.get('true_anomaly') if human_labels else None,
            'severity_rating': human_labels.get('severity_rating') if human_labels else None,
            'notes': human_labels.get('notes', '') if human_labels else ''
        }
        
        self.signal_data.append(record)
    
    def record_domain_prediction(
        self,
        text: str,
        predicted_domain: str,
        all_domain_scores: Optional[Dict[str, float]] = None,
        true_domain: Optional[str] = None
    ):
        """
        Record domain classification data for embedding fine-tuning
        
        Args:
            text: Input text (signal title + content)
            predicted_domain: System's predicted domain
            all_domain_scores: Scores for all domains (if available)
            true_domain: Human-verified correct domain
        """
        record = {
            'timestamp': datetime.now().isoformat(),
            'text': text[:2000],  # Truncate long texts
            'predicted_domain': predicted_domain,
            'domain_scores': all_domain_scores or {},
            'true_domain': true_domain,
            'needs_labeling': true_domain is None
        }
        
        self.domain_data.append(record)
    
    def record_legal_recommendation(
        self,
        signal: Dict[str, Any],
        recommendation: Dict[str, Any],
        human_rating: Optional[Dict[str, Any]] = None
    ):
        """
        Record legal recommendation for LLM fine-tuning
        
        Args:
            signal: Original signal that triggered recommendation
            recommendation: Generated legal recommendation
            human_rating: Optional human evaluation
                {
                    'quality': 4.5,  # 1-5 scale
                    'relevance': 5.0,
                    'legal_soundness': 4.0,
                    'clarity': 4.5,
                    'feedback': 'Good structure but missing enforcement details'
                }
        """
        record = {
            'timestamp': datetime.now().isoformat(),
            'signal_id': signal.get('id', 'unknown'),
            'signal_title': signal.get('title', '')[:200],
            'signal_domain': signal.get('category', 'general'),
            'signal_severity': signal.get('severity', 0.5),
            
            # Generated recommendation
            'rec_id': recommendation.get('id'),
            'rec_title': recommendation.get('title'),
            'rec_domain': recommendation.get('issue_domain'),
            'rec_law_text': recommendation.get('proposed_law'),
            'rec_rationale': recommendation.get('rationale'),
            
            # Human evaluation (to be filled later)
            'quality_rating': human_rating.get('quality') if human_rating else None,
            'relevance_rating': human_rating.get('relevance') if human_rating else None,
            'legal_soundness': human_rating.get('legal_soundness') if human_rating else None,
            'clarity_rating': human_rating.get('clarity') if human_rating else None,
            'human_feedback': human_rating.get('feedback', '') if human_rating else '',
            'needs_review': human_rating is None
        }
        
        self.legal_data.append(record)
    
    def record_rag_retrieval(
        self,
        query: str,
        domain: str,
        retrieved_docs: List[str],
        retrieval_scores: List[float],
        relevance_labels: Optional[List[bool]] = None
    ):
        """
        Record RAG retrieval for improving document ranking
        
        Args:
            query: Query text (signal content)
            domain: Domain searched
            retrieved_docs: Retrieved document texts
            retrieval_scores: Similarity scores
            relevance_labels: Human-labeled relevance [True, False, True, ...]
        """
        record = {
            'timestamp': datetime.now().isoformat(),
            'query': query[:1000],
            'domain': domain,
            'num_retrieved': len(retrieved_docs),
            'documents': [doc[:500] for doc in retrieved_docs[:5]],  # Top 5, truncated
            'scores': retrieval_scores[:5],
            'relevance_labels': relevance_labels[:5] if relevance_labels else [None] * min(5, len(retrieved_docs)),
            'needs_labeling': relevance_labels is None
        }
        
        self.rag_data.append(record)
    
    def save_session(self, session_name: Optional[str] = None):
        """
        Save collected data to disk
        
        Args:
            session_name: Optional name for this session (default: timestamp)
        """
        if session_name is None:
            session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        session_dir = self.output_dir / session_name
        session_dir.mkdir(exist_ok=True)
        
        # Save as JSON
        datasets = {
            'signals': self.signal_data,
            'domains': self.domain_data,
            'legal_recommendations': self.legal_data,
            'rag_retrievals': self.rag_data
        }
        
        for name, data in datasets.items():
            if data:
                # Save JSON
                json_path = session_dir / f"{name}.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                # Save CSV for easy viewing/labeling
                csv_path = session_dir / f"{name}.csv"
                try:
                    df = pd.DataFrame(data)
                    df.to_csv(csv_path, index=False, encoding='utf-8')
                    logger.info(f"✓ Saved {len(data)} {name} records to {csv_path}")
                except Exception as e:
                    logger.warning(f"Could not save CSV for {name}: {e}")
        
        # Save summary
        summary = {
            'session_name': session_name,
            'timestamp': datetime.now().isoformat(),
            'signal_records': len(self.signal_data),
            'domain_records': len(self.domain_data),
            'legal_records': len(self.legal_data),
            'rag_records': len(self.rag_data),
            'unlabeled_signals': sum(1 for r in self.signal_data if r['true_domain'] is None),
            'unlabeled_legal': sum(1 for r in self.legal_data if r['quality_rating'] is None),
        }
        
        summary_path = session_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"✓ Training data saved to {session_dir}")
        logger.info(f"  - {summary['signal_records']} signal records ({summary['unlabeled_signals']} need labeling)")
        logger.info(f"  - {summary['legal_records']} legal recommendations ({summary['unlabeled_legal']} need rating)")
        
        return session_dir
    
    def export_for_training(self, task: str = 'all') -> Dict[str, Path]:
        """
        Export data in training-ready format
        
        Args:
            task: 'anomaly', 'domain', 'legal', 'rag', or 'all'
        
        Returns:
            Dict of task -> file path
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = self.output_dir / f"training_export_{timestamp}"
        export_dir.mkdir(exist_ok=True)
        
        exported = {}
        
        # 1. Anomaly Detection Dataset
        if task in ['anomaly', 'all']:
            anomaly_samples = [
                {
                    'text': f"{r['title']} {r['content']}",
                    'severity': r['severity'],
                    'predicted_anomaly': r['predicted_anomaly'],
                    'true_anomaly': r['true_anomaly'],
                    'label': r['true_anomaly'] if r['true_anomaly'] is not None else r['predicted_anomaly']
                }
                for r in self.signal_data
            ]
            path = export_dir / "anomaly_detection_dataset.json"
            with open(path, 'w') as f:
                json.dump(anomaly_samples, f, indent=2)
            exported['anomaly'] = path
            logger.info(f"✓ Exported {len(anomaly_samples)} anomaly samples to {path}")
        
        # 2. Domain Classification Dataset
        if task in ['domain', 'all']:
            domain_samples = [
                {
                    'text': r['text'],
                    'label': r['true_domain'] if r['true_domain'] else r['predicted_domain']
                }
                for r in self.domain_data
            ]
            path = export_dir / "domain_classification_dataset.json"
            with open(path, 'w') as f:
                json.dump(domain_samples, f, indent=2)
            exported['domain'] = path
            logger.info(f"✓ Exported {len(domain_samples)} domain samples to {path}")
        
        # 3. Legal Recommendation Dataset (LLM fine-tuning)
        if task in ['legal', 'all']:
            legal_samples = [
                {
                    'input': f"Signal: {r['signal_title']}\nDomain: {r['signal_domain']}\nSeverity: {r['signal_severity']}",
                    'output': r['rec_law_text'],
                    'quality_score': r['quality_rating'] if r['quality_rating'] else 3.0  # Default neutral
                }
                for r in self.legal_data
            ]
            path = export_dir / "legal_recommendation_dataset.json"
            with open(path, 'w') as f:
                json.dump(legal_samples, f, indent=2)
            exported['legal'] = path
            logger.info(f"✓ Exported {len(legal_samples)} legal samples to {path}")
        
        # 4. RAG Retrieval Dataset
        if task in ['rag', 'all']:
            rag_samples = [
                {
                    'query': r['query'],
                    'domain': r['domain'],
                    'documents': r['documents'],
                    'scores': r['scores'],
                    'relevance': r['relevance_labels']
                }
                for r in self.rag_data
            ]
            path = export_dir / "rag_retrieval_dataset.json"
            with open(path, 'w') as f:
                json.dump(rag_samples, f, indent=2)
            exported['rag'] = path
            logger.info(f"✓ Exported {len(rag_samples)} RAG samples to {path}")
        
        return exported
    
    def load_human_labels(self, csv_path: str):
        """
        Load human-labeled data from CSV
        
        Args:
            csv_path: Path to CSV with human labels
        """
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"✓ Loaded {len(df)} labeled records from {csv_path}")
            
            # Update in-memory data with labels
            # (Implementation depends on CSV structure)
            return df
        except Exception as e:
            logger.error(f"Failed to load labels: {e}")
            return None
    
    def get_labeling_stats(self) -> Dict[str, Any]:
        """Get statistics on data labeling progress"""
        return {
            'total_signals': len(self.signal_data),
            'labeled_signals': sum(1 for r in self.signal_data if r['true_domain'] is not None),
            'total_legal_recs': len(self.legal_data),
            'rated_legal_recs': sum(1 for r in self.legal_data if r['quality_rating'] is not None),
            'total_rag_retrievals': len(self.rag_data),
            'labeled_rag_retrievals': sum(1 for r in self.rag_data if r['relevance_labels'] and any(r['relevance_labels'])),
            'labeling_progress': {
                'signals': f"{sum(1 for r in self.signal_data if r['true_domain'] is not None)}/{len(self.signal_data)}",
                'legal': f"{sum(1 for r in self.legal_data if r['quality_rating'] is not None)}/{len(self.legal_data)}",
                'rag': f"{sum(1 for r in self.rag_data if r['relevance_labels'])}/{len(self.rag_data)}"
            }
        }
