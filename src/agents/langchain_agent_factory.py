"""
Langchain Agent 工厂
提供统一的 Agent 创建接口，支持不同类型的 Langchain Agent
"""
from typing import Any, Dict, List, Optional
from langchain_core.tools import BaseTool

from src.models.langchain_model_adapter import create_langchain_model
from src.tools.tools import create_tools
from .langchain_agent import LangchainMedicalAgent, create_langchain_medical_agent
from .langgraph_multi_agent import LangGraphMultiAgentSystem, create_langgraph_multi_agent_system
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class LangchainAgentFactory:
    """
    Langchain Agent 工厂类
    """
    
    @staticmethod
    def create_agent(
        agent_type: str,
        model_config: Dict[str, Any],
        tools: Optional[List[BaseTool]] = None,
        system_prompt: Optional[str] = None,
        verbose: bool = False,
        **kwargs
    ):
        """
        创建 Langchain Agent
        
        Args:
            agent_type: Agent 类型 ('langchain_medical', 'langgraph_multi_agent')
            model_config: 模型配置
            tools: 工具列表
            system_prompt: 系统提示
            verbose: 是否输出详细日志
            kwargs: 其他参数
            
        Returns:
            Agent 实例
            
        Raises:
            ValueError: 如果 Agent 类型不支持
        """
        if agent_type == "langchain_medical":
            return create_langchain_medical_agent(
                model_config=model_config,
                tools=tools,
                system_prompt=system_prompt,
                verbose=verbose
            )
        
        elif agent_type == "langgraph_multi_agent":
            return create_langgraph_multi_agent_system(
                model_config=model_config,
                tools=tools,
                verbose=verbose
            )
        
        else:
            raise ValueError(f"不支持的 Agent 类型: {agent_type}")
    
    @staticmethod
    def create_default_medical_agent(
        model_config: Dict[str, Any],
        verbose: bool = False
    ) -> LangchainMedicalAgent:
        """
        创建默认的医疗 Agent
        
        Args:
            model_config: 模型配置
            verbose: 是否输出详细日志
            
        Returns:
            LangchainMedicalAgent 实例
        """
        # 创建默认工具
        tools = create_tools()
        
        # 创建 Agent
        return create_langchain_medical_agent(
            model_config=model_config,
            tools=tools,
            verbose=verbose
        )
    
    @staticmethod
    def create_multi_agent_system(
        model_config: Dict[str, Any],
        verbose: bool = False
    ) -> LangGraphMultiAgentSystem:
        """
        创建多智能体系统
        
        Args:
            model_config: 模型配置
            verbose: 是否输出详细日志
            
        Returns:
            LangGraphMultiAgentSystem 实例
        """
        # 创建默认工具
        tools = create_tools()
        
        # 创建多智能体系统
        return create_langgraph_multi_agent_system(
            model_config=model_config,
            tools=tools,
            verbose=verbose
        )
    
    @staticmethod
    def get_available_agent_types() -> List[str]:
        """
        获取可用的 Agent 类型
        
        Returns:
            Agent 类型列表
        """
        return [
            "langchain_medical",
            "langgraph_multi_agent"
        ]
    
    @staticmethod
    def get_agent_info(agent_type: str) -> Dict[str, Any]:
        """
        获取 Agent 信息
        
        Args:
            agent_type: Agent 类型
            
        Returns:
            Agent 信息字典
        """
        agent_info = {
            "langchain_medical": {
                "name": "Langchain 医疗 Agent",
                "description": "基于 Langchain 的单智能体医疗助手",
                "features": [
                    "工具调用",
                    "医疗知识问答",
                    "参数验证",
                    "错误处理"
                ],
                "suitable_for": [
                    "简单医疗咨询",
                    "工具使用",
                    "单轮对话"
                ]
            },
            "langgraph_multi_agent": {
                "name": "LangGraph 多智能体系统",
                "description": "基于 LangGraph 的多智能体协作系统",
                "features": [
                    "任务路由",
                    "多智能体协作",
                    "专业分工",
                    "响应协调"
                ],
                "suitable_for": [
                    "复杂医疗问题",
                    "多专业协作",
                    "综合分析"
                ]
            }
        }
        
        return agent_info.get(agent_type, {
            "name": "未知 Agent 类型",
            "description": "不支持的 Agent 类型",
            "features": [],
            "suitable_for": []
        })

def create_agent_from_config(config: Dict[str, Any]):
    """
    从配置文件创建 Agent
    
    Args:
        config: 配置字典，包含 agent_type, model_config 等
        
    Returns:
        Agent 实例
    """
    agent_type = config.get("agent_type", "langchain_medical")
    model_config = config.get("model_config", {})
    tools_config = config.get("tools", {})
    system_prompt = config.get("system_prompt")
    verbose = config.get("verbose", False)
    
    # 创建工具（如果需要）
    tools = None
    if tools_config.get("use_default_tools", True):
        tools = create_tools()
    
    # 创建 Agent
    return LangchainAgentFactory.create_agent(
        agent_type=agent_type,
        model_config=model_config,
        tools=tools,
        system_prompt=system_prompt,
        verbose=verbose
    ) 