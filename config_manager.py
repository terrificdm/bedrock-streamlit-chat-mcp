import json
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class ModelInfo:
    model_id: str
    max_tokens: int
    supports_multimodal: bool
    supports_video: bool
    supports_reasoning: bool
    top_k_max: int
    model_family: str
    supports_top_k: bool = True
    supports_top_p: bool = True
    supports_tools: bool = True

@dataclass
class FileTypeConfig:
    types: List[str]
    size_limit: int

class ConfigManager:
    def __init__(self, config_path: str = "models_config.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
    
    def get_model_info(self, model_name: str) -> ModelInfo:
        model_config = self.config["models"][model_name]
        return ModelInfo(**model_config)
    
    def get_all_model_names(self) -> List[str]:
        return list(self.config["models"].keys())
    
    def get_file_config(self, file_type: str) -> FileTypeConfig:
        config = self.config["file_config"][file_type]
        return FileTypeConfig(**config)
    
    def get_regions(self) -> List[str]:
        return self.config["regions"]
    
    def get_default_model(self) -> str:
        return self.config["default_model"]
    
    def get_default_region(self) -> str:
        return self.config["default_region"]
    
    def get_file_types_by_category(self) -> Dict[str, set]:
        return {
            "image": set(self.config["file_config"]["image"]["types"]),
            "document": set(self.config["file_config"]["document"]["types"]),
            "video": set(self.config["file_config"]["video"]["types"])
        }