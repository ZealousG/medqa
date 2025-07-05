#!/usr/bin/env python3
"""
医疗问答系统 Gradio 界面启动脚本
"""

import sys
import os
import socket

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.UI.gradio_interface import GradioInterface
from src.configs.configs import ModelConfig

def find_free_port(start_port=7860, max_port=7900):
    """
    寻找可用的端口
    
    Args:
        start_port: 起始端口
        max_port: 最大端口
        
    Returns:
        可用的端口号
    """
    for port in range(start_port, max_port + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    raise OSError(f"无法在 {start_port}-{max_port} 范围内找到可用端口")

def main():
    """主函数"""
    print("🏥 启动医疗问答系统...")
    print("📋 功能特性：")
    print("   • 话题式历史记录管理")
    print("   • 智能话题标题生成")
    print("   • 完整对话历史展示")
    print("   • 多模型切换支持")
    print("   • 流式输出响应")
    print("   • Markdown格式渲染")
    
    # 创建模型配置（只用默认配置）
    model_config = ModelConfig()
    # 如需修改配置，可在此赋值，例如：
    # model_config.use_api = True
    # model_config.temperature = 0.7
    # model_config.max_length = 2048
    
    # 查找可用端口
    try:
        available_port = find_free_port(7860, 7900)
        print(f"🔍 找到可用端口: {available_port}")
    except OSError as e:
        print(f"❌ 端口查找失败: {e}")
        print("🔧 尝试使用默认端口 7860...")
        available_port = 7860
    
    # 创建界面实例
    interface = GradioInterface(
        model_type="api",
        model_config=model_config,
        verbose=True
    )
    
    print("🚀 界面启动中，请稍候...")
    print("📱 界面将在浏览器中打开")
    print(f"🔗 本地访问地址: http://localhost:{available_port}")
    print("💡 使用提示：")
    print("   • 左侧面板显示话题历史，点击可查看完整对话")
    print("   • 右侧面板进行实时对话交互")
    print("   • 系统会自动为新对话生成话题标题")
    print("   • 支持流式输出和Markdown格式显示")
    
    # 启动界面
    try:
        interface.launch(
            server_name="0.0.0.0",
            server_port=available_port,
            share=False,
            debug=True,
            show_error=True
        )
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        print("🔧 尝试使用随机端口...")
        interface.launch(
            server_name="0.0.0.0",
            server_port=0,  # 0 表示使用随机可用端口
            share=False,
            debug=True,
            show_error=True
        )

if __name__ == "__main__":
    main() 