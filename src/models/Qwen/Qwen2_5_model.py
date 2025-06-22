from langchain.llms.base import LLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
from src.configs.model_config import ModelConfig
from src.utils.logger import setup_logger
from pydantic import PrivateAttr

logger = setup_logger(__name__)

class Qwen2_5_Coder(LLM):
    """基于本地 Qwen2_5-Coder 自定义 LLM 类"""
    
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None
    _model_config: ModelConfig = PrivateAttr()
    
    def __init__(self, model_config: ModelConfig):
        """
        初始化Qwen2_5模型
        
        Args:
            model_config: 模型配置，包含model_path等参数
        """
        super().__init__()
        self._model_config = model_config
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        try:
            logger.info(f"正在从本地加载模型: {self._model_config.model_path}")
            
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self._model_config.model_path, 
                use_fast=False
            )
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self._model_config.model_path, 
                torch_dtype=torch.bfloat16, 
                device_map="auto"
            )
            
            # 设置生成配置
            self.model.generation_config = GenerationConfig.from_pretrained(
                self._model_config.model_path
            )
            
            logger.info("本地模型加载完成")
            
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any) -> str:
        """
        调用模型生成回复
        
        Args:
            prompt: 输入提示
            stop: 停止词列表
            run_manager: 回调管理器
            **kwargs: 其他参数
            
        Returns:
            生成的回复文本
        """
        try:
            # 构建消息格式
            messages = [{"role": "user", "content": prompt}]
            
            # 应用聊天模板
            input_ids = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # 编码输入
            model_inputs = self.tokenizer(
                [input_ids], 
                return_tensors="pt"
            ).to('cuda')
            
            # 生成回复
            generated_ids = self.model.generate(
                model_inputs.input_ids, 
                attention_mask=model_inputs['attention_mask'], 
                max_new_tokens=self._model_config.max_length,
                temperature=self._model_config.temperature,
                do_sample=True
            )
            
            # 解码输出
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids 
                in zip(model_inputs.input_ids, generated_ids)
            ]
            
            response = self.tokenizer.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            return response
            
        except Exception as e:
            logger.error(f"模型推理失败: {str(e)}")
            raise
    
    @property
    def _llm_type(self) -> str:
        """返回模型类型标识"""
        return "Qwen2_5_Coder"
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            "model_type": "Qwen2_5_Coder",
            "model_path": self._model_config.model_path,
            "device": self._model_config.device,
            "temperature": self._model_config.temperature,
            "max_length": self._model_config.max_length
        }