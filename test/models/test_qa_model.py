"""
QA模型测试文件
测试QAModel类的各种功能
"""
import pytest
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from langchain_core.language_models import BaseLLM
from langchain_openai import ChatOpenAI

from src.models.QA_model import QAModel, APIModelStrategy, LocalModelStrategy, create_qa_model
from src.configs.configs import ModelConfig
from src.models.Qwen.Qwen2_5_model import Qwen2_5_Coder


class TestAPIModelStrategy:
    """测试API模型策略"""
    
    def test_create_model(self):
        """测试创建API模型"""
        config = ModelConfig()
        config.volc_model_name = "qwen-turbo"
        config.volc_api_base = "https://api.volcengine.com"
        config.volc_api_key = "test_key"
        config.temperature = 0.7
        config.max_length = 1000
        
        strategy = APIModelStrategy()
        
        with patch('src.models.QA_model.ChatOpenAI') as mock_chat_openai:
            mock_model = Mock(spec=BaseLLM)
            mock_chat_openai.return_value = mock_model
            
            result = strategy.create_model(config)
            
            mock_chat_openai.assert_called_once_with(
                model="qwen-turbo",
                openai_api_base="https://api.volcengine.com",
                openai_api_key="test_key",
                temperature=0.7,
                max_tokens=1000
            )
            assert result == mock_model


class TestLocalModelStrategy:
    """测试本地模型策略"""
    
    def test_create_model(self):
        """测试创建本地模型"""
        config = ModelConfig()
        config.model_path = "/path/to/model"
        
        strategy = LocalModelStrategy()
        
        with patch('src.models.QA_model.Qwen2_5_Coder') as mock_qwen:
            mock_model = Mock(spec=BaseLLM)
            mock_qwen.return_value = mock_model
            
            result = strategy.create_model(config)
            
            mock_qwen.assert_called_once_with(config)
            assert result == mock_model


class TestQAModel:
    """测试QA模型类"""
    
    def setup_method(self):
        """设置测试方法"""
        self.config = ModelConfig()
        self.config.volc_model_name = "qwen-turbo"
        self.config.volc_api_base = "https://api.volcengine.com"
        self.config.volc_api_key = "test_key"
        self.config.model_path = "/path/to/model"
        self.config.use_api = True
        self.config.temperature = 0.7
        self.config.max_length = 1000
    
    def test_init(self):
        """测试初始化"""
        qa_model = QAModel(self.config)
        assert qa_model.model_config == self.config
        assert "api" in qa_model.strategies
        assert "local" in qa_model.strategies
        assert isinstance(qa_model.model_cache, dict)
    
    def test_init_without_config(self):
        """测试无配置初始化"""
        qa_model = QAModel()
        assert isinstance(qa_model.model_config, ModelConfig)
    
    @patch('src.models.QA_model.ChatOpenAI')
    def test_create_model_api(self, mock_chat_openai):
        """测试创建API模型"""
        qa_model = QAModel(self.config)
        mock_model = Mock(spec=BaseLLM)
        mock_chat_openai.return_value = mock_model
        
        result = qa_model.create_model("api")
        
        assert result == mock_model
        assert len(qa_model.model_cache) == 1
    
    @patch('src.models.QA_model.Qwen2_5_Coder')
    def test_create_model_local(self, mock_qwen):
        """测试创建本地模型"""
        qa_model = QAModel(self.config)
        mock_model = Mock(spec=BaseLLM)
        mock_qwen.return_value = mock_model
        
        result = qa_model.create_model("local")
        
        assert result == mock_model
        assert len(qa_model.model_cache) == 1
    
    def test_create_model_invalid_type(self):
        """测试创建无效模型类型"""
        qa_model = QAModel(self.config)
        
        with pytest.raises(ValueError, match="不支持的模型类型"):
            qa_model.create_model("invalid")
    
    @patch('src.models.QA_model.ChatOpenAI')
    def test_create_model_auto_select_api(self, mock_chat_openai):
        """测试自动选择API模型"""
        self.config.use_api = True
        qa_model = QAModel(self.config)
        mock_model = Mock(spec=BaseLLM)
        mock_chat_openai.return_value = mock_model
        
        result = qa_model.create_model()
        
        assert result == mock_model
    
    @patch('src.models.QA_model.Qwen2_5_Coder')
    def test_create_model_auto_select_local(self, mock_qwen):
        """测试自动选择本地模型"""
        self.config.use_api = False
        qa_model = QAModel(self.config)
        mock_model = Mock(spec=BaseLLM)
        mock_qwen.return_value = mock_model
        
        result = qa_model.create_model()
        
        assert result == mock_model
    
    @patch('src.models.QA_model.ChatOpenAI')
    def test_model_caching(self, mock_chat_openai):
        """测试模型缓存功能"""
        qa_model = QAModel(self.config)
        mock_model = Mock(spec=BaseLLM)
        mock_chat_openai.return_value = mock_model
        
        # 第一次创建
        result1 = qa_model.create_model("api")
        # 第二次创建（应该使用缓存）
        result2 = qa_model.create_model("api")
        
        assert result1 == result2
        assert mock_chat_openai.call_count == 1  # 只调用一次
        assert len(qa_model.model_cache) == 1
    
    def test_clear_cache(self):
        """测试清除缓存"""
        qa_model = QAModel(self.config)
        qa_model.model_cache["test"] = "model"
        
        qa_model.clear_cache()
        
        assert len(qa_model.model_cache) == 0
    
    def test_get_cached_models(self):
        """测试获取缓存模型信息"""
        qa_model = QAModel(self.config)
        mock_model = Mock()
        mock_model.__class__.__name__ = "TestModel"
        qa_model.model_cache["test_key"] = mock_model
        
        result = qa_model.get_cached_models()
        
        assert result == {"test_key": "TestModel"}


class TestCreateQAModelFunction:
    """测试便捷函数create_qa_model"""
    
    def setup_method(self):
        """设置测试方法"""
        self.config = ModelConfig()
        self.config.volc_model_name = "qwen-turbo"
        self.config.volc_api_base = "https://api.volcengine.com"
        self.config.volc_api_key = "test_key"
        self.config.use_api = True
    
    @patch('src.models.QA_model.ChatOpenAI')
    def test_create_qa_model_with_config(self, mock_chat_openai):
        """测试使用配置创建模型"""
        mock_model = Mock(spec=BaseLLM)
        mock_chat_openai.return_value = mock_model
        
        result = create_qa_model("api", self.config)
        
        assert result == mock_model
    
    @patch('src.models.QA_model.ChatOpenAI')
    def test_create_qa_model_without_config(self, mock_chat_openai):
        """测试不使用配置创建模型"""
        mock_model = Mock(spec=BaseLLM)
        mock_chat_openai.return_value = mock_model
        
        result = create_qa_model("api")
        
        assert result == mock_model
    
    @patch('src.models.QA_model.Qwen2_5_Coder')
    def test_create_qa_model_local(self, mock_qwen):
        """测试创建本地模型"""
        self.config.use_api = False
        mock_model = Mock(spec=BaseLLM)
        mock_qwen.return_value = mock_model
        
        result = create_qa_model("local", self.config)
        
        assert result == mock_model


if __name__ == "__main__":
    pytest.main([__file__]) 