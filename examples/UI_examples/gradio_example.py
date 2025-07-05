#!/usr/bin/env python3
"""
Gradio 界面使用示例
"""

import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.UI.gradio_interface import GradioInterface
from src.configs.configs import ModelConfig

def example_basic_usage():
    """基本使用示例"""
    print("=== 基本使用示例 ===")
    
    # 创建界面实例，使用默认配置
    interface = GradioInterface(
        model_type="api",
        verbose=True
    )
    
    # 启动界面
    interface.launch(
        server_port=7860,
        share=False
    )

def example_with_custom_config():
    """自定义配置示例（仅赋值已定义属性）"""
    print("=== 自定义配置示例 ===")
    
    # 创建自定义模型配置
    model_config = ModelConfig()
    model_config.use_api = True
    model_config.temperature = 0.8
    model_config.max_length = 1024
    model_config.top_p = 0.9
    
    # 创建界面实例
    interface = GradioInterface(
        model_type="api",
        model_config=model_config,
        verbose=True
    )
    
    # 启动界面
    interface.launch(
        server_port=7861,
        share=False
    )

def example_local_model():
    """本地模型示例（仅赋值已定义属性）"""
    print("=== 本地模型示例 ===")
    
    # 创建本地模型配置
    model_config = ModelConfig()
    model_config.use_api = False
    model_config.model_path = "./models/local_model"
    model_config.device = "cuda"
    model_config.temperature = 0.7
    
    # 创建界面实例
    interface = GradioInterface(
        model_type="local",
        model_config=model_config,
        verbose=True
    )
    
    # 启动界面
    interface.launch(
        server_port=7862,
        share=False
    )

if __name__ == "__main__":
    print("🏥 医疗问答系统 Gradio 界面示例")
    print("请选择要运行的示例:")
    print("1. 基本使用示例")
    print("2. 自定义配置示例")
    print("3. 本地模型示例")
    
    choice = input("请输入选择 (1-3): ").strip()
    
    if choice == "1":
        example_basic_usage()
    elif choice == "2":
        example_with_custom_config()
    elif choice == "3":
        example_local_model()
    else:
        print("无效选择，运行基本示例...")
        example_basic_usage() 