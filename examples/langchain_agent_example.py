"""
Langchain Agent 使用示例
"""
import yaml
import sys
import os
from src.configs.model_config import ModelConfig

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.agents.langchain_agent_factory import LangchainAgentFactory, create_agent_from_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

cfg = ModelConfig()

def example_single_agent():
    """
    单智能体示例
    """
    print("=== 单智能体示例 ===")
    
    # 模型配置
    model_config = {
        "model_name": cfg.model_path,
        "device": cfg.device,
        "max_length": cfg.max_length,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p
    }
    
    # 创建 Agent
    agent = LangchainAgentFactory.create_default_medical_agent(
        model_config=model_config,
        verbose=True
    )
    
    # 测试查询
    queries = [
        "我最近经常头痛，可能是什么原因？",
        "请计算一下 BMI，我的体重是 70kg，身高是 1.75m",
        "高血压患者应该注意什么？"
    ]
    
    for query in queries:
        print(f"\n用户查询: {query}")
        result = agent.run(query)
        print(f"Agent 响应: {result['response']}")
        print(f"元数据: {result['metadata']}")

def example_multi_agent():
    """
    多智能体系统示例
    """
    print("\n=== 多智能体系统示例 ===")
    
    # 模型配置
    model_config = {
        "model_name": cfg.model_path,
        "device": cfg.device,
        "max_length": cfg.max_length,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p
    }
    
    # 创建多智能体系统
    system = LangchainAgentFactory.create_multi_agent_system(
        model_config=model_config,
        verbose=True
    )
    
    # 测试查询
    queries = [
        "我最近胸痛，可能是什么疾病？",
        "糖尿病患者的治疗方案有哪些？",
        "如何预防心血管疾病？"
    ]
    
    for query in queries:
        print(f"\n用户查询: {query}")
        result = system.run(query)
        print(f"系统响应: {result['response']}")
        print(f"各 Agent 响应: {result['metadata']['agent_responses']}")
        print(f"当前 Agent: {result['metadata']['current_agent']}")

def example_from_config():
    """
    从 ModelConfig 创建 Agent 示例（不再读取 yaml）
    """
    print("\n=== 从 ModelConfig 创建 Agent 示例 ===")
    
    # 直接用 ModelConfig 实例参数
    model_config = {
        "model_name": cfg.model_path,
        "device": cfg.device,
        "max_length": cfg.max_length,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p
    }
    agent = create_agent_from_config({
        "agent_type": "langchain_medical",
        "model_config": model_config,
        "verbose": cfg.verbose
    })
    
    query = "我最近失眠，有什么建议吗？"
    print(f"用户查询: {query}")
    result = agent.run(query)
    print(f"Agent 响应: {result['response']}")

def example_agent_comparison():
    """
    Agent 类型对比示例
    """
    print("\n=== Agent 类型对比 ===")
    
    # 模型配置
    model_config = {
        "model_name": cfg.model_path,
        "device": cfg.device,
        "max_length": cfg.max_length,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p
    }
    
    # 获取可用 Agent 类型
    agent_types = LangchainAgentFactory.get_available_agent_types()
    
    for agent_type in agent_types:
        print(f"\n--- {agent_type} ---")
        info = LangchainAgentFactory.get_agent_info(agent_type)
        print(f"名称: {info['name']}")
        print(f"描述: {info['description']}")
        print(f"特性: {', '.join(info['features'])}")
        print(f"适用场景: {', '.join(info['suitable_for'])}")
        
        # 创建 Agent 并测试
        try:
            agent = LangchainAgentFactory.create_agent(
                agent_type=agent_type,
                model_config=model_config,
                verbose=False
            )
            
            query = "我最近头痛，可能是什么原因？"
            result = agent.run(query)
            print(f"测试查询: {query}")
            print(f"响应长度: {len(result['response'])} 字符")
            print(f"Agent 类型: {result['metadata']['agent_type']}")
            
        except Exception as e:
            print(f"创建失败: {str(e)}")

def main():
    """
    主函数
    """
    print("Langchain Agent 示例程序")
    print("=" * 50)
    
    try:
        # 单智能体示例
        # example_single_agent()
        
        # 多智能体系统示例
        example_multi_agent()
        
        # 从 ModelConfig 创建示例
        example_from_config()
        
        # Agent 类型对比
        example_agent_comparison()
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"示例程序执行失败: {str(e)}")
        print(f"错误: {str(e)}")

if __name__ == "__main__":
    main() 