import sys; sys.setrecursionlimit(5000)

"""
基于 Langchain 的 Agent 实现
"""
from typing import Any, Dict, List, Optional
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseLLM

from src.models.langchain_model_adapter import LangchainModelAdapter
from src.tools.tools import create_tools
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class LangchainMedicalAgent:
    """
    基于 Langchain 的医疗 Agent
    """
    
    def __init__(
        self,
        model: LangchainModelAdapter,
        tools: Optional[List[BaseTool]] = None,
        system_prompt: Optional[str] = None,
        verbose: bool = False
    ):
        """
        初始化 Langchain 医疗 Agent
        
        Args:
            model: Langchain 兼容的模型
            tools: 工具列表
            system_prompt: 系统提示
            verbose: 是否输出详细日志
        """
        self.model = model
        self.tools = tools or create_tools()
        self.verbose = verbose
        
        # 默认系统提示
        self.system_prompt = system_prompt or (
            "你是一个专业的医疗助手，可以回答医疗相关问题并能使用工具来获取更准确的信息。"
            "在回答医疗问题时，请遵循以下原则：\n"
            "1. 基于可靠的医学知识提供准确的信息\n"
            "2. 不要做出诊断、处方或治疗建议，而是提供一般的医学知识\n"
            "3. 对于紧急情况，建议用户立即就医\n"
            "4. 当你不确定时，坦率承认并鼓励用户咨询专业医生\n"
            "5. 提供实用的一般健康信息和预防措施\n"
            "请用清晰、易懂的语言回答问题，避免使用过多的专业术语。"
        )
        
        # 创建 Agent
        self.agent_executor = self._create_agent()
    
    def _create_agent(self) -> AgentExecutor:
        """
        创建 Langchain Agent
        
        Returns:
            AgentExecutor 实例
        """
        # 创建提示模板
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # 创建 Agent
        agent = create_openai_tools_agent(
            llm=self.model,
            tools=self.tools,
            prompt=prompt
        )
        
        # 创建 Agent 执行器
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=self.verbose,
            handle_parsing_errors=True,
            max_iterations=5
        )
        
        return agent_executor
    
    def run(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        运行 Agent
        
        Args:
            query: 用户查询
            chat_history: 聊天历史
            
        Returns:
            执行结果
        """
        try:
            # 转换聊天历史格式
            messages = self._convert_chat_history(chat_history) if chat_history else []
            
            # 执行 Agent
            result = self.agent_executor.invoke({
                "input": query,
                "chat_history": messages
            })
            
            return {
                "response": result.get("output", ""),
                "chat_history": chat_history or [],
                "metadata": {
                    "agent_type": "langchain",
                    "tools_used": result.get("intermediate_steps", []),
                    "iterations": len(result.get("intermediate_steps", []))
                }
            }
            
        except Exception as e:
            logger.error(f"Agent 执行失败: {str(e)}")
            return {
                "response": f"处理失败: {str(e)}",
                "chat_history": chat_history or [],
                "metadata": {
                    "error": str(e),
                    "agent_type": "langchain"
                }
            }
    
    def _convert_chat_history(self, chat_history: List[Dict[str, str]]) -> List[BaseMessage]:
        """
        转换聊天历史格式
        
        Args:
            chat_history: 原始聊天历史
            
        Returns:
            Langchain 消息格式
        """
        messages = []
        
        for message in chat_history:
            if message["role"] == "user":
                messages.append(HumanMessage(content=message["content"]))
            elif message["role"] == "assistant":
                messages.append(AIMessage(content=message["content"]))
        
        return messages
    
    def add_tool(self, tool: BaseTool) -> None:
        """
        添加工具
        
        Args:
            tool: 要添加的工具
        """
        self.tools.append(tool)
        # 重新创建 Agent 以包含新工具
        self.agent_executor = self._create_agent()
    
    def get_tools(self) -> List[BaseTool]:
        """
        获取所有工具
        
        Returns:
            工具列表
        """
        return self.tools

def create_langchain_medical_agent(
    model_config: Dict[str, Any],
    tools: Optional[List[BaseTool]] = None,
    system_prompt: Optional[str] = None,
    verbose: bool = False
) -> LangchainMedicalAgent:
    """
    创建 Langchain 医疗 Agent
    
    Args:
        model_config: 模型配置
        tools: 工具列表
        system_prompt: 系统提示
        verbose: 是否输出详细日志
        
    Returns:
        LangchainMedicalAgent 实例
    """
    from src.models.langchain_model_adapter import create_langchain_model
    
    # 创建模型
    model = create_langchain_model(model_config)
    
    # 创建 Agent
    agent = LangchainMedicalAgent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        verbose=verbose
    )
    
    return agent 