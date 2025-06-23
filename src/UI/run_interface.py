#!/usr/bin/env python3
"""
医疗问答系统 Gradio 界面启动脚本
"""

import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.UI.gradio_interface import GradioInterface
from src.configs.model_config import ModelConfig

def main():
    """主函数"""
    print("🏥 启动医疗问答系统...")
    
    # 创建模型配置（只用默认配置）
    model_config = ModelConfig()
    # 如需修改配置，可在此赋值，例如：
    # model_config.use_api = True
    # model_config.temperature = 0.7
    # model_config.max_length = 2048
    
    # 创建界面实例
    interface = GradioInterface(
        model_type="api",
        model_config=model_config,
        verbose=True
    )
    
    print("🚀 界面启动中，请稍候...")
    print("📱 界面将在浏览器中打开")
    print("🔗 本地访问地址: http://localhost:7860")
    
    # 启动界面
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True
    )

if __name__ == "__main__":
    main() 