"""
QA Agent 使用示例
统一使用ModelConfig配置
"""
import sys
import os
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.agents.QA_agent import QA_Agent
from src.configs.configs import ModelConfig
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def check_environment():
    """检查环境变量配置"""
    print("=== 环境变量检查 ===")
    
    volc_api_key = os.getenv("VOLC_API_KEY")
    volc_api_base = os.getenv("VOLC_API_BASE")
    volc_model = os.getenv("VOLC_MODEL")
    
    print(f"VOLC_API_KEY: {'已设置' if volc_api_key else '未设置'}")
    print(f"VOLC_API_BASE: {'已设置' if volc_api_base else '未设置'}")
    print(f"VOLC_MODEL: {volc_model or '未设置'}")
    
    if not volc_api_key or not volc_api_base or not volc_model:
        print("警告: 火山引擎API配置不完整，请设置以下环境变量:")
        print("export VOLC_API_KEY='your-volc-api-key'")
        print("export VOLC_API_BASE='your-volc-api-base'")
        print("export VOLC_MODEL='your-model-name'")
        return False
    
    return True

def example_basic_usage():
    """基本使用示例"""
    print("=== 基本使用示例 ===")
    
    # 使用默认配置创建Agent
    agent = QA_Agent(verbose=True)
    
    # 运行查询
    result = agent.run("什么是高血压？")
    print(f"查询: {result['query']}")
    print(f"响应: {result['response']}")
    print(f"模型类型: {result['model_type']}")

def example_custom_config():
    """自定义配置示例"""
    print("\n=== 自定义配置示例 ===")
    
    # 创建自定义配置
    config = ModelConfig()
    config.use_api = True  # 使用火山引擎API
    config.temperature = 0.5
    config.max_length = 1024
    
    # 使用自定义配置创建Agent
    agent = QA_Agent(model_config=config, verbose=True)
    
    # 查看模型信息
    model_info = agent.get_model_info()
    print(f"模型信息: {model_info}")
    
    # 运行查询
    result = agent.run("糖尿病的症状有哪些？")
    print(f"响应: {result['response']}")

def example_model_switching():
    """模型切换示例"""
    print("\n=== 模型切换示例 ===")
    
    # 创建Agent（使用默认API模型）
    agent = QA_Agent(verbose=True)
    
    query = "头痛的原因有哪些？"
    
    # 使用API模型
    print("使用API模型:")
    result1 = agent.run(query)
    print(f"响应: {result1['response'][:100]}...")
    
    # 切换到本地模型
    print("\n切换到本地模型:")
    agent.switch_model("local")
    result2 = agent.run(query)
    print(f"响应: {result2['response'][:100]}...")

def example_with_tools():
    """使用工具的示例"""
    print("\n=== 使用工具的示例 ===")
    
    # 创建Agent
    agent = QA_Agent(verbose=True)
    
    # 查看可用工具
    tools = agent.get_available_tools()
    print(f"可用工具: {tools}")
    
    # 运行需要计算的查询
    result = agent.run("请计算 25 * 36 的结果")
    print(f"查询: {result['query']}")
    print(f"响应: {result['response']}")

def example_chat_history():
    """聊天历史示例"""
    print("\n=== 聊天历史示例 ===")
    
    agent = QA_Agent(verbose=True)
    
    # 模拟聊天历史
    chat_history = [
        {"role": "user", "content": "什么是高血压？"},
        {"role": "assistant", "content": "高血压是指血压持续升高的疾病..."},
        {"role": "user", "content": "它的症状有哪些？"}
    ]
    
    # 运行带历史记录的查询
    result = agent.run("如何预防高血压？", chat_history)
    print(f"查询: {result['query']}")
    print(f"响应: {result['response']}")

def example_direct_model():
    """直接使用模型实例示例"""
    print("\n=== 直接使用模型实例示例 ===")
    
    from langchain_openai import ChatOpenAI
    
    # 直接创建LangChain模型（使用火山引擎配置）
    llm = ChatOpenAI(
        model=os.getenv("VOLC_MODEL"),
        openai_api_base=os.getenv("VOLC_API_BASE"),
        openai_api_key=os.getenv("VOLC_API_KEY"),
        temperature=0.7
    )
    
    # 直接传递给Agent
    agent = QA_Agent(model=llm, verbose=True)
    
    result = agent.run("头痛的原因有哪些？")
    print(f"查询: {result['query']}")
    print(f"响应: {result['response']}")
    print(f"模型类型: {result['model_type']}")

def example_error_handling():
    """错误处理示例"""
    print("\n=== 错误处理示例 ===")
    
    # 测试无效的模型类型
    try:
        agent = QA_Agent(model_type="invalid_type")
    except Exception as e:
        print(f"预期的错误: {e}")
    
    # 测试无效的配置
    try:
        config = ModelConfig()
        config.model_path = "/invalid/path"
        agent = QA_Agent(model_config=config)
        result = agent.run("测试查询")
    except Exception as e:
        print(f"配置错误: {e}")

if __name__ == "__main__":
    # 检查环境变量
    if not check_environment():
        print("请先设置正确的环境变量后再运行示例")
        exit(1)
    
    # 运行所有示例
    example_basic_usage()
    example_custom_config()
    # example_model_switching()
    example_with_tools()
    example_chat_history()
    example_direct_model()
    example_error_handling()
    
    print("\n=== 示例完成 ===") 