
from productive_agents.llm import ChatGPT, Gemini, vLLM, AzureOpenAIServerModel, OllamaModel
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import yaml
import os

def load_openai_key_from_config() -> Optional[str]:
    """Load OpenAI API key from private_config.yaml"""
    config_paths = [
        "configs/private_config.yaml",
        "../configs/private_config.yaml", 
        "../../configs/private_config.yaml",
        os.path.expanduser("~/Projects/productive-agents-submission/configs/private_config.yaml")
    ]
    
    for config_path in config_paths:
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    if config and 'openai_key' in config:
                        return config['openai_key']
        except Exception as e:
            continue
    
    # Try environment variable as fallback
    return os.getenv('OPENAI_API_KEY')

def load_ollama_base_url_from_config() -> str:
    """Load Ollama base URL from private_config.yaml or env var, defaulting to http://localhost:11434/v1"""
    config_paths = [
        "configs/private_config.yaml",
        "../configs/private_config.yaml", 
        "../../configs/private_config.yaml",
        os.path.expanduser("~/Projects/productive-agents-submission/configs/private_config.yaml")
    ]
    
    for config_path in config_paths:
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    if config and 'ollama_base_url' in config:
                        return config['ollama_base_url']
        except Exception as e:
            continue
    
    return os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434/v1')

class LLMManager:
    """Manages LLM initialization and inference."""
    
    @staticmethod
    def create_llm(model_name: str, key: Optional[str] = None, system_message: str = "", lora_name: Optional[str] = None, backend: Optional[str] = None):
        """Create appropriate LLM instance based on model name."""
        model_name_lower = model_name.lower()
        ollama_url = load_ollama_base_url_from_config()
        env_backend = os.getenv('LLM_BACKEND', '').lower()

        # 1. Explicit Ollama specification or prefix
        if (
            model_name_lower.startswith('ollama/')
            or model_name_lower.startswith('ollama:')
            or backend == 'ollama'
            or env_backend == 'ollama'
        ):
            return OllamaModel(model_name, base_url=ollama_url, system_message=system_message)

        # 2. OpenAI / Azure GPT models
        elif any(pattern in model_name_lower for pattern in ['gpt', 'o1', 'o3', 'o4']):
            if model_name.startswith('azure/'):
                model_name = model_name.split('/')[1]
            
            openai_key = load_openai_key_from_config() or key
            if not openai_key:
                raise ValueError(
                    f"OpenAI API key not found for model '{model_name}'. "
                    "Place your key in configs/private_config.yaml or set OPENAI_API_KEY. "
                    "If you want to use a local Ollama model, specify '--model_name ollama/<model>' (e.g. ollama/llama3.1:8b)."
                )
            
            return ChatGPT(model_name, openai_key, system_message)

        # 3. Gemini models
        elif 'gemini' in model_name_lower:
            return Gemini(model_name, key, system_message)

        # 4. Explicit vLLM prefix
        elif model_name_lower.startswith('vllm/'):
            clean_name = model_name[len('vllm/'):]
            return vLLM(clean_name, system_message, lora_name=lora_name)

        # 5. Open-source models or tags (llama, qwen, mistral, deepseek, phi, or tag with colon like :8b)
        elif any(k in model_name_lower for k in ['llama', 'qwen', 'mistral', 'deepseek', 'phi', 'gemma']) or ':' in model_name:
            return OllamaModel(model_name, base_url=ollama_url, system_message=system_message)

        # 6. Fallback (try OllamaModel first)
        else:
            return OllamaModel(model_name, base_url=ollama_url, system_message=system_message)


# dataclass for managing output from agent.forward
@dataclass
class LLMOutput:
    """Holds the output from an agent's forward method."""
    action: Any
    response: str
    metadata: Dict[str, Any] = None

