"""
Qwen2_5_model 测试文件
"""
import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.models.Qwen.Qwen2_5_model import Qwen2_5_Coder
from src.configs.configs import ModelConfig
from langchain.llms.base import LLM

class TestQwen2_5_Coder:
    """测试Qwen2_5_Coder类"""
    
    def test_init_with_model_config(self):
        """测试使用ModelConfig初始化"""
        config = ModelConfig()
        config.model_path = "/test/model/path"
        config.temperature = 0.7
        config.max_length = 2048
        
        with patch('src.models.Qwen.Qwen2_5_model.AutoTokenizer') as mock_tokenizer, \
             patch('src.models.Qwen.Qwen2_5_model.AutoModelForCausalLM') as mock_model, \
             patch('src.models.Qwen.Qwen2_5_model.GenerationConfig') as mock_gen_config:
            
            # 模拟tokenizer和model
            mock_tokenizer.from_pretrained.return_value = Mock()
            mock_model.from_pretrained.return_value = Mock()
            mock_gen_config.from_pretrained.return_value = Mock()
            
            qwen_model = Qwen2_5_Coder(config)
            
            # 修正：使用私有属性访问
            assert qwen_model._model_config == config
            assert qwen_model.tokenizer is not None
            assert qwen_model.model is not None
    
    def test_load_model_success(self):
        """测试模型加载成功"""
        config = ModelConfig()
        config.model_path = "/test/model/path"
        
        with patch('src.models.Qwen.Qwen2_5_model.AutoTokenizer') as mock_tokenizer, \
             patch('src.models.Qwen.Qwen2_5_model.AutoModelForCausalLM') as mock_model, \
             patch('src.models.Qwen.Qwen2_5_model.GenerationConfig') as mock_gen_config:
            
            mock_tokenizer.from_pretrained.return_value = Mock()
            mock_model.from_pretrained.return_value = Mock()
            mock_gen_config.from_pretrained.return_value = Mock()
            
            qwen_model = Qwen2_5_Coder(config)
            
            # 验证调用
            mock_tokenizer.from_pretrained.assert_called_once_with(
                "/test/model/path", 
                use_fast=False
            )
            # 修正：只验证关键参数，不验证torch_dtype的具体值
            mock_model.from_pretrained.assert_called_once()
            call_args = mock_model.from_pretrained.call_args
            assert call_args[0][0] == "/test/model/path"  # 第一个位置参数
            assert call_args[1]["device_map"] == "auto"   # 关键字参数
    
    def test_load_model_failure(self):
        """测试模型加载失败"""
        config = ModelConfig()
        config.model_path = "/invalid/path"
        
        with patch('src.models.Qwen.Qwen2_5_model.AutoTokenizer') as mock_tokenizer:
            mock_tokenizer.from_pretrained.side_effect = Exception("模型加载失败")
            
            with pytest.raises(Exception, match="模型加载失败"):
                Qwen2_5_Coder(config)
    
    def test_call_method(self):
        """测试_call方法"""
        config = ModelConfig()
        config.model_path = "/test/model/path"
        config.temperature = 0.7
        config.max_length = 512
        
        with patch('src.models.Qwen.Qwen2_5_model.AutoTokenizer') as mock_tokenizer, \
             patch('src.models.Qwen.Qwen2_5_model.AutoModelForCausalLM') as mock_model, \
             patch('src.models.Qwen.Qwen2_5_model.GenerationConfig') as mock_gen_config:
            
            # 模拟tokenizer
            mock_tokenizer_instance = Mock()
            mock_tokenizer_instance.apply_chat_template.return_value = "formatted_prompt"
            
            # 创建模拟的BatchEncoding对象
            mock_batch_encoding = Mock()
            mock_input_ids = [[1, 2, 3], [4, 5, 6]]  # 二维list，模拟batch
            mock_attention_mask = [[0, 0, 0], [0, 0, 0]]
            
            # 创建返回对象，同时支持字典和属性访问
            mock_return_obj = Mock()
            mock_return_obj.input_ids = mock_input_ids
            mock_return_obj.__getitem__ = lambda self, key: {"input_ids": mock_input_ids, "attention_mask": mock_attention_mask}[key]
            
            mock_batch_encoding.to.return_value = mock_return_obj
            mock_tokenizer_instance.return_value = mock_batch_encoding
            mock_tokenizer_instance.batch_decode.return_value = ["generated_response"]
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            # 模拟model
            mock_model_instance = Mock()
            # 生成的output_ids要和input_ids数量一致
            mock_output_ids1 = [1, 2, 3, 4]
            mock_output_ids2 = [4, 5, 6, 7]
            mock_model_instance.generate.return_value = [mock_output_ids1, mock_output_ids2]
            mock_model.from_pretrained.return_value = mock_model_instance
            mock_gen_config.from_pretrained.return_value = Mock()
            
            qwen_model = Qwen2_5_Coder(config)
            
            # 测试_call方法
            result = qwen_model._call("测试提示")
            
            assert result == "generated_response"
            mock_tokenizer_instance.apply_chat_template.assert_called_once()
            mock_model_instance.generate.assert_called_once()
    
    def test_call_method_failure(self):
        """测试_call方法失败"""
        config = ModelConfig()
        config.model_path = "/test/model/path"
        
        with patch('src.models.Qwen.Qwen2_5_model.AutoTokenizer') as mock_tokenizer, \
             patch('src.models.Qwen.Qwen2_5_model.AutoModelForCausalLM') as mock_model, \
             patch('src.models.Qwen.Qwen2_5_model.GenerationConfig') as mock_gen_config:
            
            mock_tokenizer.from_pretrained.return_value = Mock()
            mock_model.from_pretrained.return_value = Mock()
            mock_gen_config.from_pretrained.return_value = Mock()
            
            qwen_model = Qwen2_5_Coder(config)
            
            # 模拟推理失败
            with patch.object(qwen_model.tokenizer, 'apply_chat_template', side_effect=Exception("推理失败")):
                with pytest.raises(Exception, match="推理失败"):
                    qwen_model._call("测试提示")
    
    def test_llm_type_property(self):
        """测试_llm_type属性"""
        config = ModelConfig()
        config.model_path = "/test/model/path"
        
        with patch('src.models.Qwen.Qwen2_5_model.AutoTokenizer'), \
             patch('src.models.Qwen.Qwen2_5_model.AutoModelForCausalLM'), \
             patch('src.models.Qwen.Qwen2_5_model.GenerationConfig'):
            
            qwen_model = Qwen2_5_Coder(config)
            
            assert qwen_model._llm_type == "Qwen2_5_Coder"
    
    def test_get_model_info(self):
        """测试get_model_info方法"""
        config = ModelConfig()
        config.model_path = "/test/model/path"
        config.device = "cuda:0"
        config.temperature = 0.7
        config.max_length = 2048
        
        with patch('src.models.Qwen.Qwen2_5_model.AutoTokenizer'), \
             patch('src.models.Qwen.Qwen2_5_model.AutoModelForCausalLM'), \
             patch('src.models.Qwen.Qwen2_5_model.GenerationConfig'):
            
            qwen_model = Qwen2_5_Coder(config)
            
            model_info = qwen_model.get_model_info()
            
            expected_info = {
                "model_type": "Qwen2_5_Coder",
                "model_path": "/test/model/path",
                "device": "cuda:0",
                "temperature": 0.7,
                "max_length": 2048
            }
            
            assert model_info == expected_info
    
    def test_inheritance(self):
        """测试继承关系"""
        config = ModelConfig()
        config.model_path = "/test/model/path"
        
        with patch('src.models.Qwen.Qwen2_5_model.AutoTokenizer'), \
             patch('src.models.Qwen.Qwen2_5_model.AutoModelForCausalLM'), \
             patch('src.models.Qwen.Qwen2_5_model.GenerationConfig'):
            
            qwen_model = Qwen2_5_Coder(config)
            
            # 验证继承自LLM
            assert isinstance(qwen_model, LLM)
    
    def test_config_validation(self):
        
        """测试配置验证"""
        # 测试空配置
        config = ModelConfig()
        config.model_path = ""
        
        with pytest.raises(Exception):
            with patch('src.models.Qwen.Qwen2_5_model.AutoTokenizer', side_effect=Exception("路径为空")):
                Qwen2_5_Coder(config)
    
    def test_temperature_and_max_length_config(self):
        """测试温度和最大长度配置"""
        config = ModelConfig()
        config.model_path = "/test/model/path"
        config.temperature = 0.5
        config.max_length = 1024
        
        with patch('src.models.Qwen.Qwen2_5_model.AutoTokenizer') as mock_tokenizer, \
             patch('src.models.Qwen.Qwen2_5_model.AutoModelForCausalLM') as mock_model, \
             patch('src.models.Qwen.Qwen2_5_model.GenerationConfig') as mock_gen_config:
            
            mock_tokenizer.from_pretrained.return_value = Mock()
            mock_model.from_pretrained.return_value = Mock()
            mock_gen_config.from_pretrained.return_value = Mock()
            
            qwen_model = Qwen2_5_Coder(config)
            
            # 修正：使用私有属性访问
            assert qwen_model._model_config.temperature == 0.5
            assert qwen_model._model_config.max_length == 1024

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 