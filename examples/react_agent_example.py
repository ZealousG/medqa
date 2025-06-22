"""
ReAct 智能体使用示例
"""
import sys
import os
from typing import Dict, Any

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.langchain_model_adapter import create_langchain_model
from src.agents.react_agent import create_react_agent
from src.tools.calculator_tool import CalculatorTool
from src.tools.medical_assessment_tool import MedicalAssessmentTool
from src.tools.search_tool import SearchTool
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def create_react_agent_example():
    """
    创建 ReAct 智能体示例
    """
    # 模型配置
    model_config = {
        "model_type": "qwen",
        "model_name": "Qwen/Qwen2-7B-Instruct",
        "device": "cuda",
        "max_length": 2048,
        "temperature": 0.7
    }
    
    # 创建模型
    model = create_langchain_model(model_config)
    
    # 创建工具
    tools = {
        "计算工具": CalculatorTool(),
        "医疗评估": MedicalAssessmentTool(),
        "搜索工具": SearchTool()
    }
    
    # 创建 ReAct 智能体
    react_agent = create_react_agent(
        model=model,
        tools=tools,
        name="医疗推理智能体",
        description="专门用于处理复杂医疗问题的推理智能体，能够进行多步分析和工具调用",
        max_steps=8,
        verbose=True
    )
    
    return react_agent

def run_react_agent_example():
    """
    运行 ReAct 智能体示例
    """
    # 创建智能体
    agent = create_react_agent_example()
    
    # 示例任务
    tasks = [
        "请分析一个45岁男性患者，体重80kg，身高175cm，血压140/90mmHg，总胆固醇220mg/dL，HDL 45mg/dL，是否吸烟，无糖尿病史。请评估其心血管疾病风险并计算BMI。",
        
        "一个60岁女性患者出现胸痛、气短、出汗等症状，请分析可能的诊断，并考虑需要哪些进一步的检查。",
        
        "请研究一下高血压患者的饮食建议，包括钠盐摄入、钾的补充、以及DASH饮食模式的效果。"
    ]
    
    print("=== ReAct 智能体示例 ===\n")
    
    for i, task in enumerate(tasks, 1):
        print(f"任务 {i}: {task}")
        print("-" * 80)
        
        try:
            # 执行任务
            result = agent.run(task)
            
            print(f"响应:\n{result['response']}")
            print(f"\n元数据:")
            print(f"- 执行步骤数: {result['metadata']['steps_executed']}")
            print(f"- 使用的工具: {[tool['tool'] for tool in result['metadata']['tools_used']]}")
            
        except Exception as e:
            print(f"执行失败: {str(e)}")
        
        print("\n" + "=" * 80 + "\n")

def run_multi_agent_with_react_example():
    """
    运行包含 ReAct 智能体的多智能体系统示例
    """
    from src.agents.langgraph_multi_agent import create_langgraph_multi_agent_system
    
    # 模型配置
    model_config = {
        "model_type": "qwen",
        "model_name": "Qwen/Qwen2-7B-Instruct",
        "device": "cuda",
        "max_length": 2048,
        "temperature": 0.7
    }
    
    # 创建多智能体系统
    multi_agent_system = create_langgraph_multi_agent_system(
        model_config=model_config,
        verbose=True
    )
    
    # 复杂推理任务
    complex_task = "请综合分析一个50岁男性糖尿病患者的整体健康状况，包括心血管风险评估、血糖控制建议、并发症预防措施，并制定个性化的健康管理计划。"
    
    print("=== 多智能体系统 + ReAct 智能体示例 ===\n")
    print(f"任务: {complex_task}")
    print("-" * 80)
    
    try:
        # 执行任务
        result = multi_agent_system.run(complex_task)
        
        print(f"最终响应:\n{result['response']}")
        print(f"\n各智能体响应:")
        for agent_name, response in result['metadata'].get('agent_responses', {}).items():
            print(f"- {agent_name}: {response[:100]}...")
            
    except Exception as e:
        print(f"执行失败: {str(e)}")

if __name__ == "__main__":
    # 运行 ReAct 智能体示例
    run_react_agent_example()
    
    # 运行多智能体系统示例
    # run_multi_agent_with_react_example() 