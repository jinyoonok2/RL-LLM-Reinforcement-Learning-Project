"""
Prompt template loader for different model types.
Loads external YAML templates for clean separation of prompts and code.
"""

import yaml
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class PromptLoader:
    """Loads and manages prompt templates for different model types."""
    
    def __init__(self, prompts_dir: str = "configs/prompts"):
        self.prompts_dir = Path(prompts_dir)
        self._templates = {}
        self._model_template_cache = {}  # Cache model -> template mapping
        self._load_templates()
    
    def _load_templates(self):
        """Load all prompt templates from YAML files."""
        if not self.prompts_dir.exists():
            logger.warning(f"Prompts directory not found: {self.prompts_dir}")
            return
        
        for template_file in self.prompts_dir.glob("*.yaml"):
            try:
                with open(template_file, 'r') as f:
                    template_data = yaml.safe_load(f)
                
                template_name = template_file.stem
                self._templates[template_name] = template_data
                logger.info(f"Loaded prompt template: {template_name}")
                
            except Exception as e:
                logger.error(f"Failed to load template {template_file}: {e}")
    
    def get_template_for_model(self, model_name: str) -> Dict[str, Any]:
        """Get appropriate prompt template for model."""
        # Check cache first
        if model_name in self._model_template_cache:
            return self._model_template_cache[model_name]
        
        model_name_lower = model_name.lower()
        
        # Find compatible template
        for template_name, template_data in self._templates.items():
            compatible_models = template_data.get('compatible_models', [])
            for model_pattern in compatible_models:
                if model_pattern.lower() in model_name_lower:
                    logger.info(f"Using prompt template '{template_name}' for model '{model_name}'")
                    self._model_template_cache[model_name] = template_data
                    return template_data
        
        # Default fallback to llama template
        if 'llama' in self._templates:
            logger.warning(f"No specific template for '{model_name}', using llama default")
            template_data = self._templates['llama']
            self._model_template_cache[model_name] = template_data
            return template_data
        
        raise ValueError(f"No compatible prompt template found for model: {model_name}")
    
    def format_prompt(self, model_name: str, context: str, question: str) -> str:
        """Format prompt using appropriate template for model."""
        template_data = self.get_template_for_model(model_name)
        
        # Handle different template formats
        if 'system_message' in template_data and 'user_template' in template_data:
            # Chat template format (Phi-3 style)
            system_msg = template_data['system_message']
            user_msg = template_data['user_template'].format(context=context, question=question)
            
            if 'template' in template_data:
                return template_data['template'].format(
                    system_message=system_msg,
                    user_message=user_msg
                )
            else:
                return f"<|system|>\n{system_msg}<|end|>\n<|user|>\n{user_msg}<|end|>\n<|assistant|>\n"
        
        elif 'template' in template_data:
            # Simple template format (Llama style)
            return template_data['template'].format(context=context, question=question)
        
        else:
            raise ValueError(f"Invalid template format in {template_data}")

# Global instance
_prompt_loader = None

def get_prompt_loader() -> PromptLoader:
    """Get global prompt loader instance."""
    global _prompt_loader
    if _prompt_loader is None:
        _prompt_loader = PromptLoader()
    return _prompt_loader