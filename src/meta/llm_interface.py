"""
Local LLM Interface using Fine-Tuned Mistral 7B via HuggingFace Transformers
Uses QLoRA fine-tuned model for AI ethics law generation

No API keys needed! Runs entirely on your GPU.
"""

import torch
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from config.settings import settings

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("transformers/peft not available. Install: pip install transformers peft")

logger = logging.getLogger(__name__)


class LocalLLMInterface:
    """Interface for fine-tuned local LLM - 100% Free!"""
    
    def __init__(self, model: str = None):
        self.model_name = model or settings.HUGGINGFACE_MODEL
        self.finetuned_path = settings.FINETUNED_MODEL_PATH
        self.use_finetuned = self.finetuned_path.exists() and settings.USE_HUGGINGFACE
        
        self.model = None
        self.tokenizer = None
        self.available = False
        
        # Try to load the fine-tuned model
        if TRANSFORMERS_AVAILABLE and self.use_finetuned:
            self.available = self._load_finetuned_model()
        
        if self.available:
            model_type = "fine-tuned" if self.use_finetuned else "base"
            logger.info(f"✓ Loaded {model_type} Mistral 7B model")
        else:
            logger.warning("⚠ LLM not available - using template-based generation")
            if not TRANSFORMERS_AVAILABLE:
                logger.warning("Install transformers: pip install transformers peft bitsandbytes")
    
    def _load_finetuned_model(self) -> bool:
        """Load fine-tuned model with LoRA adapters"""
        try:
            logger.info("Loading fine-tuned Mistral 7B model...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
            
            # 4-bit quantization config for inference
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            
            # Load base model in 4-bit
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            )
            
            # Load fine-tuned LoRA adapters
            self.model = PeftModel.from_pretrained(
                base_model,
                str(self.finetuned_path)
            )
            
            logger.info("✓ Fine-tuned model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load fine-tuned model: {e}")
            return False
    
    def generate_agent_spec(self, anomaly_context: Dict[str, Any]) -> Dict:
        """
        Use local LLM to generate agent specification
        Falls back to templates if LLM unavailable
        """
        
        if not self.available:
            return self._template_fallback(anomaly_context)
        
        try:
            prompt = self._build_agent_generation_prompt(anomaly_context)
            response = self._generate_text(prompt, max_tokens=500)
            
            # Parse LLM response
            agent_spec = self._parse_llm_response(response)
            
            if agent_spec:
                logger.info("✓ Generated agent spec using fine-tuned LLM")
                return agent_spec
            else:
                logger.warning("Failed to parse LLM response, using template")
                return self._template_fallback(anomaly_context)
                
        except Exception as e:
            logger.error(f"Error calling fine-tuned LLM: {e}")
            return self._template_fallback(anomaly_context)
    
    def collaborative_decision(self, signal_data: Dict[str, Any]) -> Dict:
        """
        Use LLM to synthesize collaborative decision
        Enhances algorithmic voting with natural language reasoning
        """
        
        if not self.available:
            return self._algorithmic_fallback(signal_data)
        
        try:
            prompt = self._build_collaboration_prompt(signal_data)
            response = self._generate_text(prompt, max_tokens=500)
            
            # Extract decision reasoning
            decision = self._parse_collaboration_response(response)
            
            logger.info("✓ Enhanced collaboration with fine-tuned LLM reasoning")
            return decision
            
        except Exception as e:
            logger.error(f"Error in LLM collaboration: {e}")
            return self._algorithmic_fallback(signal_data)
    
    def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        """
        Generic text generation method for law proposals and other text
        
        Args:
            prompt: The prompt to generate from
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text string
        """
        if not self.available:
            raise Exception("LLM not available - fine-tuned model not loaded")
        
        return self._generate_text(prompt, max_tokens)
    
    def _generate_text(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate text using fine-tuned model"""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )
            
            # Move to model's device
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode output (skip input prompt)
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            return generated_text.strip()
            
        except Exception as e:
            raise Exception(f"LLM generation failed: {e}")
    
    def _build_agent_generation_prompt(self, context: Dict) -> str:
        """Build prompt for agent generation"""
        
        signal = context.get('signal', {})
        category = signal.get('category', 'unknown')
        severity = signal.get('severity', 0.5)
        unexplained = context.get('unexplained_factors', [])
        
        prompt = f"""You are an AI system architect designing specialized ethical monitoring agents.

CONTEXT:
An anomaly has been detected in the {category} domain with severity {severity:.2f}.

Unexplained factors:
{chr(10).join(f"- {factor}" for factor in unexplained)}

TASK:
Design a specialized agent to handle this gap. Provide:

1. Agent Name: A descriptive name (e.g., "Privacy Protection Specialist")
2. Capabilities: List 3-5 specific capabilities (comma-separated)
3. Tools: List 4-6 tools this agent needs (comma-separated)
4. Focus: One sentence describing the agent's main responsibility

FORMAT YOUR RESPONSE EXACTLY AS:
Agent Name: [name]
Capabilities: [cap1, cap2, cap3]
Tools: [tool1, tool2, tool3]
Focus: [one sentence focus]

Be specific and technical. Focus on {category} domain."""

        return prompt
    
    def _build_collaboration_prompt(self, signal_data: Dict) -> str:
        """Build prompt for collaborative decision"""
        
        signal = signal_data.get('signal', {})
        assessments = signal_data.get('assessments', [])
        
        prompt = f"""You are synthesizing multiple AI agent assessments for ethical oversight.

SIGNAL:
Category: {signal.get('category', 'unknown')}
Content: {signal.get('content', '')[:200]}
Severity: {signal.get('severity', 0.5):.2f}

AGENT ASSESSMENTS:
"""
        
        for i, assessment in enumerate(assessments, 1):
            prompt += f"\nAgent {i} ({assessment.get('agent_name', 'Unknown')}): "
            prompt += f"Score {assessment.get('score', 0):.2f}, "
            prompt += f"Confidence {assessment.get('coverage', 0):.2f}\n"
        
        prompt += """
TASK:
1. Analyze the disagreements and agreements between agents
2. Identify the key ethical concern
3. Recommend an action (acceptable/monitor/investigate/urgent)
4. Explain your reasoning in 2-3 sentences

FORMAT:
Recommendation: [acceptable/monitor/investigate/urgent]
Reasoning: [2-3 sentences explaining why]
Key Concern: [one main ethical issue identified]"""

        return prompt
    
    def _parse_llm_response(self, response: str) -> Optional[Dict]:
        """Parse LLM response into agent specification"""
        
        try:
            lines = response.strip().split('\n')
            spec = {}
            
            for line in lines:
                if ':' not in line:
                    continue
                    
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                
                if 'name' in key:
                    spec['name'] = value
                elif 'capabilities' in key:
                    spec['capabilities'] = [c.strip() for c in value.split(',')]
                elif 'tools' in key:
                    spec['tools'] = [t.strip() for t in value.split(',')]
                elif 'focus' in key:
                    spec['focus'] = value
            
            # Validate we got the essentials
            if spec.get('name') and spec.get('capabilities'):
                return spec
            
            return None
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return None
    
    def _parse_collaboration_response(self, response: str) -> Dict:
        """Parse LLM collaboration response"""
        
        result = {
            'recommendation': 'monitor',
            'reasoning': response[:200],
            'key_concern': 'Unknown',
            'llm_enhanced': True
        }
        
        try:
            lines = response.strip().split('\n')
            
            for line in lines:
                if ':' not in line:
                    continue
                    
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                
                if 'recommendation' in key:
                    rec = value.lower()
                    if any(r in rec for r in ['urgent', 'immediate']):
                        result['recommendation'] = 'urgent_investigation'
                    elif 'investigate' in rec:
                        result['recommendation'] = 'investigate'
                    elif 'monitor' in rec:
                        result['recommendation'] = 'monitor'
                    elif 'acceptable' in rec:
                        result['recommendation'] = 'acceptable'
                
                elif 'reasoning' in key:
                    result['reasoning'] = value
                
                elif 'concern' in key:
                    result['key_concern'] = value
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing collaboration response: {e}")
            return result
    
    def _template_fallback(self, context: Dict) -> Dict:
        """Fallback to template-based generation (all 7 domains)"""
        
        signal = context.get('signal', {})
        category = signal.get('category', 'general')
        
        templates = {
            'bias': {
                'name': 'Bias Detection Specialist',
                'capabilities': ['bias_detection', 'fairness_analysis', 'discrimination_monitoring'],
                'tools': ['statistical_analysis', 'demographic_breakdown', 'fairness_metrics'],
                'focus': 'identifying and measuring bias in AI systems'
            },
            'privacy': {
                'name': 'Privacy Protection Agent',
                'capabilities': ['privacy_monitoring', 'data_protection', 'pii_detection'],
                'tools': ['data_flow_analysis', 'encryption_check', 'privacy_audit'],
                'focus': 'protecting user privacy and data'
            },
            'transparency': {
                'name': 'Transparency Oversight Agent',
                'capabilities': ['explainability_analysis', 'disclosure_monitoring', 'interpretability_check'],
                'tools': ['model_inspection', 'shap_analysis', 'audit_trail'],
                'focus': 'ensuring AI systems are transparent and explainable'
            },
            'accountability': {
                'name': 'Accountability Compliance Agent',
                'capabilities': ['audit_monitoring', 'liability_analysis', 'responsibility_tracking'],
                'tools': ['audit_trail', 'compliance_check', 'incident_tracker'],
                'focus': 'establishing clear accountability for AI outcomes'
            },
            'safety': {
                'name': 'Safety Assurance Agent',
                'capabilities': ['harm_prevention', 'risk_assessment', 'safety_testing'],
                'tools': ['risk_analysis', 'incident_detection', 'failsafe_monitor'],
                'focus': 'preventing AI systems from causing harm'
            },
            'security': {
                'name': 'Security Compliance Agent',
                'capabilities': ['vulnerability_detection', 'threat_analysis', 'security_monitoring'],
                'tools': ['threat_scanner', 'access_control_audit', 'penetration_test'],
                'focus': 'securing AI systems against cyber threats'
            },
            'general': {
                'name': 'General Ethics Oversight Agent',
                'capabilities': ['ethics_analysis', 'governance_monitoring', 'policy_compliance'],
                'tools': ['policy_analysis', 'ethics_review', 'compliance_check'],
                'focus': 'overseeing general AI ethics and governance'
            }
        }
        
        template = templates.get(category, templates['general'])
        return template
    
    def _algorithmic_fallback(self, signal_data: Dict) -> Dict:
        """Fallback to pure algorithmic decision"""
        
        return {
            'recommendation': 'monitor',
            'reasoning': 'Algorithmic consensus (no LLM available)',
            'key_concern': 'Standard assessment',
            'llm_enhanced': False
        }


# Alias for backwards compatibility
LLMInterface = LocalLLMInterface