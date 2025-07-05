"""
QA模型类
实现两种模型策略：API模式和本地模型模式
统一使用ModelConfig配置
"""
import os
from typing import Dict, Any, Optional, Union

from langchain_core.language_models import BaseLLM
from langchain_openai import ChatOpenAI

from .Qwen.Qwen2_5_model import Qwen2_5_Coder
from src.configs.configs import ModelConfig
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class APIModelStrategy:
    """API模型策略"""
    
    def create_model(self, config: ModelConfig) -> BaseLLM:
        """创建API模型"""
        # 使用火山引擎API
        llm = ChatOpenAI(
            model=config.volc_model_name or os.getenv("VOLC_MODEL", "qwen-turbo"),
            openai_api_base=config.volc_api_base or os.getenv("VOLC_API_BASE"),
            openai_api_key=config.volc_api_key or os.getenv("VOLC_API_KEY"),
            temperature=config.temperature,
            max_tokens=config.max_length
        )
        
        logger.info(f"API模型创建成功: {config.volc_model_name or os.getenv('VOLC_MODEL', 'qwen-turbo')}")
        return llm

class LocalModelStrategy:
    """本地模型策略"""
    
    def create_model(self, config: ModelConfig) -> BaseLLM:
        """创建本地模型"""
        # 使用Qwen2_5_Coder创建本地模型
        base_model = Qwen2_5_Coder(config)
        
        logger.info(f"本地模型创建成功: {config.model_path}")
        return base_model

class QAModel:
    """
    QA模型类
    支持API模式和本地模型模式
    统一使用ModelConfig配置
    """
    
    def __init__(self, model_config: Optional[ModelConfig] = None):
        self.model_config = model_config or ModelConfig()
        self.strategies = {
            "api": APIModelStrategy(),
            "local": LocalModelStrategy()
        }
        self.model_cache = {}
    
    def create_model(self, model_type: str = None) -> BaseLLM:
        """
        创建模型实例
        
        Args:
            model_type: 模型类型 ("api", "local")，如果为None则根据config.use_api自动选择
            
        Returns:
            BaseLLM 实例
        """
        # 如果没有指定model_type，根据配置自动选择
        if model_type is None:
            model_type = "api" if self.model_config.use_api else "local"
        
        if model_type not in self.strategies:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 生成缓存键
        cache_key = f"{model_type}_{self.model_config.use_api}_{self.model_config.model_path}"
        
        # 检查缓存
        if cache_key in self.model_cache:
            logger.info(f"使用缓存的模型: {cache_key}")
            return self.model_cache[cache_key]
        
        # 创建模型
        strategy = self.strategies[model_type]
        model = strategy.create_model(self.model_config)
        
        # 缓存模型
        self.model_cache[cache_key] = model
        
        return model
    
    def clear_cache(self):
        """清除模型缓存"""
        self.model_cache.clear()
        logger.info("模型缓存已清除")
    
    def get_cached_models(self) -> Dict[str, str]:
        """获取缓存的模型信息"""
        return {key: type(model).__name__ for key, model in self.model_cache.items()}

# 全局QA模型实例
qa_model = QAModel()

def create_qa_model(model_type: str = None, model_config: Optional[ModelConfig] = None) -> BaseLLM:
    """
    便捷函数：创建QA模型
    
    Args:
        model_type: 模型类型，如果为None则根据config.use_api自动选择
        model_config: 模型配置（可选）
        
    Returns:
        BaseLLM 实例
    """
    if model_config:
        qa_model_instance = QAModel(model_config)
        return qa_model_instance.create_model(model_type)
    else:
        return qa_model.create_model(model_type)
