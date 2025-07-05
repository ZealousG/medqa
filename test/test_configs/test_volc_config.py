#!/usr/bin/env python3
"""
测试火山引擎API配置
"""
import os
import sys
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_volc_config():
    """测试火山引擎配置"""
    print("=== 火山引擎API配置测试 ===")
    
    # 检查环境变量
    volc_api_key = os.getenv("VOLC_API_KEY")
    volc_api_base = os.getenv("VOLC_API_BASE")
    volc_model = os.getenv("VOLC_MODEL")
    
    print(f"VOLC_API_KEY: {'已设置' if volc_api_key else '未设置'}")
    print(f"VOLC_API_BASE: {'已设置' if volc_api_base else '未设置'}")
    print(f"VOLC_MODEL: {volc_model or '未设置'}")
    
    if not volc_api_key or not volc_api_base or not volc_model:
        print("\n❌ 配置不完整！请设置以下环境变量:")
        print("export VOLC_API_KEY='your-volc-api-key'")
        print("export VOLC_API_BASE='your-volc-api-base'")
        print("export VOLC_MODEL='your-model-name'")
        return False
    
    print("\n✅ 配置完整！")
    
    # 测试创建模型
    try:
        from src.configs.configs import ModelConfig
        from src.models.QA_model import create_qa_model
        
        # 创建配置
        config = ModelConfig()
        config.use_api = True
        config.volc_api_key = volc_api_key
        config.volc_api_base = volc_api_base
        config.volc_model_name = volc_model
        
        # 创建模型
        print("正在创建模型...")
        model = create_qa_model("api", config)
        print(f"✅ 模型创建成功: {type(model).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return False

if __name__ == "__main__":
    success = test_volc_config()
    if success:
        print("\n🎉 配置测试通过！可以运行示例了。")
    else:
        print("\n💥 配置测试失败！请检查环境变量设置。") 