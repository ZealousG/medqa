"""
Qwen2_5_model 使用示例
"""
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.models.Qwen.Qwen2_5_model import Qwen2_5_Coder
from src.configs.model_config import ModelConfig

def example_basic_usage():
    """基本使用示例"""
    print("=== Qwen2_5_Coder 基本使用示例 ===")
    
    # 创建配置
    config = ModelConfig()
    config.model_path = "/mnt/d/data/LLMs/Qwen/Qwen2.5-1.5B"  # 根据实际路径修改
    config.temperature = 0.7
    config.max_length = 512
    config.device = "cuda:0"
    
    try:
        # 创建模型实例
        print("正在创建Qwen2_5_Coder模型...")
        qwen_model = Qwen2_5_Coder(config)
        
        # 测试推理
        prompt = "请解释什么是人工智能？"
        print(f"输入提示: {prompt}")
        
        response = qwen_model._call(prompt)
        print(f"模型回复: {response}")
        
        # 获取模型信息
        model_info = qwen_model.get_model_info()
        print(f"模型信息: {model_info}")
        
    except Exception as e:
        print(f"错误: {e}")
        print("注意: 请确保模型路径正确且有足够的GPU内存")

def example_with_different_configs():
    """不同配置示例"""
    print("\n=== 不同配置示例 ===")
    
    # 配置1: 低温度，短回复
    config1 = ModelConfig()
    config1.model_path = "/mnt/d/data/LLMs/Qwen/Qwen2.5-1.5B"
    config1.temperature = 0.1
    config1.max_length = 256
    
    # 配置2: 高温度，长回复
    config2 = ModelConfig()
    config2.model_path = "/mnt/d/data/LLMs/Qwen/Qwen2.5-1.5B"
    config2.temperature = 0.9
    config2.max_length = 1024
    
    print("配置1: 低温度(0.1), 短回复(256)")
    print("配置2: 高温度(0.9), 长回复(1024)")
    
    # 注意: 实际使用时需要确保模型已加载
    print("注意: 实际运行需要确保模型路径正确")

def example_error_handling():
    """错误处理示例"""
    print("\n=== 错误处理示例 ===")
    
    # 测试无效路径
    config = ModelConfig()
    config.model_path = "/invalid/path"
    
    try:
        qwen_model = Qwen2_5_Coder(config)
    except Exception as e:
        print(f"预期的错误 (无效路径): {e}")
    
    # 测试空路径
    config.model_path = ""
    
    try:
        qwen_model = Qwen2_5_Coder(config)
    except Exception as e:
        print(f"预期的错误 (空路径): {e}")

if __name__ == "__main__":
    # 运行示例
    example_basic_usage()
    example_with_different_configs()
    example_error_handling()
    
    print("\n=== 示例完成 ===")
    print("注意: 要实际运行模型推理，请确保:")
    print("1. 模型路径正确")
    print("2. 有足够的GPU内存")
    print("3. 已安装必要的依赖包 (transformers, torch等)") 