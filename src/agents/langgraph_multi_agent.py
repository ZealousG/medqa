"""
基于 LangGraph 的多智能体系统
"""
from typing import Any, Dict, List, Optional, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import BaseTool

from src.models.langchain_model_adapter import LangchainModelAdapter, create_langchain_model
from src.tools.tools import create_tools
from src.agents.react_agent import ReActAgent, create_react_agent
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class AgentState(TypedDict):
    """Agent 状态定义"""
    messages: Annotated[List[BaseMessage], "聊天消息列表"]
    current_agent: Annotated[str, "当前执行的 Agent"]
    agent_responses: Annotated[Dict[str, str], "各 Agent 的响应"]
    final_response: Annotated[Optional[str], "最终响应"]
    metadata: Annotated[Dict[str, Any], "元数据"]

class LangGraphMultiAgentSystem:
    """
    基于 LangGraph 的多智能体系统
    """
    
    def __init__(
        self,
        model: LangchainModelAdapter,
        tools: Optional[List[BaseTool]] = None,
        verbose: bool = False
    ):
        """
        初始化多智能体系统
        
        Args:
            model: Langchain 兼容的模型
            tools: 工具列表
            verbose: 是否输出详细日志
        """
        self.model = model
        self.tools = tools or create_tools()
        self.verbose = verbose
        
        # 创建专业 Agent
        self.specialized_agents = self._create_specialized_agents()
        
        # 创建 ReAct 智能体
        self.react_agent = self._create_react_agent()
        
        # 创建工作流图
        self.workflow = self._create_workflow()
    
    def _create_specialized_agents(self) -> Dict[str, Any]:
        """
        创建专业 Agent
        
        Returns:
            专业 Agent 字典
        """
        agents = {}
        
        # 诊断 Agent
        diagnosis_prompt = (
            "你是一个专注于医疗诊断的Agent。你的职责是分析用户描述的症状和体征，提供可能的诊断和鉴别诊断。"
            "你应该提出关键问题来完善诊断，并解释不同诊断的可能性和依据。"
            "在回答时要保持客观谨慎，避免确定性诊断，而是提供可能性分析和下一步建议。"
        )
        
        agents["诊断Agent"] = {
            "prompt": diagnosis_prompt,
            "tools": self.tools
        }
        
        # 治疗 Agent
        treatment_prompt = (
            "你是一个专注于医疗治疗的Agent。你的职责是提供关于治疗方法、药物使用和治疗计划的信息。"
            "你应该基于用户的情况，提供关于常规治疗方案、药物选择、可能的副作用和治疗效果的信息。"
            "在回答时要平衡治疗的效果和风险，提供循证医学的证据支持，同时考虑治疗的个体化。"
        )
        
        agents["治疗Agent"] = {
            "prompt": treatment_prompt,
            "tools": self.tools
        }
        
        # 预防 Agent
        prevention_prompt = (
            "你是一个专注于疾病预防和健康维护的Agent。你的职责是提供关于健康生活方式、疾病预防和健康监测的信息。"
            "你应该基于用户的情况，提供关于饮食、运动、生活习惯和预防措施的建议，以减少疾病风险和提高生活质量。"
            "在回答时要强调预防的重要性，提供实用的健康建议，同时考虑用户的实际情况和可行性。"
        )
        
        agents["预防Agent"] = {
            "prompt": prevention_prompt,
            "tools": self.tools
        }
        
        return agents
    
    def _create_react_agent(self) -> ReActAgent:
        """
        创建 ReAct 智能体
        
        Returns:
            ReActAgent 实例
        """
        # 将工具列表转换为字典格式
        tools_dict = {}
        for tool in self.tools:
            tools_dict[tool.name] = tool
        
        return create_react_agent(
            model=self.model,
            tools=tools_dict,
            name="推理智能体",
            description="用于处理复杂的医疗推理任务，能够进行多步思考和行动来解决问题",
            max_steps=5,
            verbose=self.verbose
        )
    
    def _create_workflow(self) -> StateGraph:
        """
        创建工作流图
        
        Returns:
            StateGraph 实例
        """
        # 创建状态图
        workflow = StateGraph(AgentState)
        
        # 添加节点
        workflow.add_node("task_router", self._task_router_node)
        workflow.add_node("diagnosis_agent", self._create_agent_node("诊断Agent"))
        workflow.add_node("treatment_agent", self._create_agent_node("治疗Agent"))
        workflow.add_node("prevention_agent", self._create_agent_node("预防Agent"))
        workflow.add_node("react_agent", self._react_agent_node)
        workflow.add_node("coordinator", self._coordinator_node)
        
        # 设置入口点
        workflow.set_entry_point("task_router")
        
        # 添加条件边
        workflow.add_conditional_edges(
            "task_router",
            self._route_to_agents,
            {
                "diagnosis_agent": "diagnosis_agent",
                "treatment_agent": "treatment_agent",
                "prevention_agent": "prevention_agent",
                "react_agent": "react_agent",
                "coordinator": "coordinator"
            }
        )
        
        # 添加边到协调器
        workflow.add_edge("diagnosis_agent", "coordinator")
        workflow.add_edge("treatment_agent", "coordinator")
        workflow.add_edge("prevention_agent", "coordinator")
        workflow.add_edge("react_agent", "coordinator")
        
        # 设置结束点
        workflow.add_edge("coordinator", END)
        
        return workflow.compile()
    
    def _task_router_node(self, state: AgentState) -> AgentState:
        """
        任务路由节点
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态
        """
        # 获取用户查询
        user_message = state["messages"][-1].content if state["messages"] else ""
        
        # 扩展的路由逻辑，包含 ReAct 智能体
        if any(keyword in user_message for keyword in ["复杂", "推理", "多步", "分析", "研究"]):
            state["current_agent"] = "react_agent"
        elif any(keyword in user_message for keyword in ["症状", "诊断", "疾病", "疼痛"]):
            state["current_agent"] = "diagnosis_agent"
        elif any(keyword in user_message for keyword in ["治疗", "药物", "用药", "手术"]):
            state["current_agent"] = "treatment_agent"
        elif any(keyword in user_message for keyword in ["预防", "保健", "生活方式", "饮食"]):
            state["current_agent"] = "prevention_agent"
        else:
            # 默认使用诊断 Agent
            state["current_agent"] = "diagnosis_agent"
        
        return state
    
    def _react_agent_node(self, state: AgentState) -> AgentState:
        """
        ReAct 智能体节点
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态
        """
        try:
            # 获取用户查询
            user_message = state["messages"][-1].content if state["messages"] else ""
            
            # 执行 ReAct 智能体
            result = self.react_agent.run(user_message, self._convert_chat_history_for_react(state["messages"]))
            
            # 保存响应
            state["agent_responses"]["推理智能体"] = result.get("response", "")
            
            if self.verbose:
                logger.info(f"ReAct智能体响应: {result.get('response', '')}")
                logger.info(f"ReAct智能体元数据: {result.get('metadata', {})}")
            
        except Exception as e:
            logger.error(f"ReAct智能体执行失败: {str(e)}")
            state["agent_responses"]["推理智能体"] = f"执行失败: {str(e)}"
        
        return state
    
    def _create_agent_node(self, agent_name: str):
        """
        创建 Agent 节点
        
        Args:
            agent_name: Agent 名称
            
        Returns:
            节点函数
        """
        def agent_node(state: AgentState) -> AgentState:
            """Agent 节点函数"""
            try:
                # 获取 Agent 配置
                agent_config = self.specialized_agents[agent_name]
                
                # 创建临时 Agent 执行器
                from langchain.agents import AgentExecutor, create_openai_tools_agent
                from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", agent_config["prompt"]),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ])
                
                agent = create_openai_tools_agent(
                    llm=self.model,
                    tools=agent_config["tools"],
                    prompt=prompt
                )
                
                agent_executor = AgentExecutor(
                    agent=agent,
                    tools=agent_config["tools"],
                    verbose=self.verbose,
                    handle_parsing_errors=True,
                    max_iterations=3
                )
                
                # 执行 Agent
                user_message = state["messages"][-1].content if state["messages"] else ""
                result = agent_executor.invoke({
                    "input": user_message,
                    "chat_history": state["messages"][:-1] if len(state["messages"]) > 1 else []
                })
                
                # 保存响应
                state["agent_responses"][agent_name] = result.get("output", "")
                
                if self.verbose:
                    logger.info(f"{agent_name} 响应: {result.get('output', '')}")
                
            except Exception as e:
                logger.error(f"{agent_name} 执行失败: {str(e)}")
                state["agent_responses"][agent_name] = f"执行失败: {str(e)}"
            
            return state
        
        return agent_node
    
    def _coordinator_node(self, state: AgentState) -> AgentState:
        """
        协调器节点
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态
        """
        try:
            # 获取所有 Agent 响应
            responses = state["agent_responses"]
            
            if not responses:
                state["final_response"] = "很抱歉，没有获得有效的响应。"
                return state
            
            # 如果只有一个响应，直接使用
            if len(responses) == 1:
                agent_name, response = list(responses.items())[0]
                state["final_response"] = f"【{agent_name}】的回答：\n{response}"
                return state
            
            # 多个响应时，创建协调提示
            coordination_prompt = "请整合以下多个专业Agent的回答，提供一个连贯、一致的最终回答：\n\n"
            
            for agent_name, response in responses.items():
                coordination_prompt += f"【{agent_name}】:\n{response}\n\n"
            
            coordination_prompt += "请整合以上回答，提供最终一致的响应。"
            
            # 使用模型进行协调
            from langchain_core.prompts import ChatPromptTemplate
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "你是一个医疗多Agent系统的协调员。你的职责是综合多个专业Agent的回答，提供最终的一致性响应。"),
                ("human", coordination_prompt)
            ])
            
            chain = prompt | self.model
            result = chain.invoke({})
            
            state["final_response"] = result.content
            
        except Exception as e:
            logger.error(f"协调器执行失败: {str(e)}")
            state["final_response"] = f"协调失败: {str(e)}"
        
        return state
    
    def _route_to_agents(self, state: AgentState) -> str:
        """
        路由到相应的 Agent
        
        Args:
            state: 当前状态
            
        Returns:
            下一个节点名称
        """
        current_agent = state.get("current_agent", "diagnosis_agent")
        
        if current_agent == "diagnosis_agent":
            return "diagnosis_agent"
        elif current_agent == "treatment_agent":
            return "treatment_agent"
        elif current_agent == "prevention_agent":
            return "prevention_agent"
        elif current_agent == "react_agent":
            return "react_agent"
        else:
            return "coordinator"
    
    def run(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        运行多智能体系统
        
        Args:
            query: 用户查询
            chat_history: 聊天历史
            
        Returns:
            执行结果
        """
        try:
            # 转换聊天历史
            messages = self._convert_chat_history(chat_history) if chat_history else []
            messages.append(HumanMessage(content=query))
            
            # 初始化状态
            initial_state = AgentState(
                messages=messages,
                current_agent="",
                agent_responses={},
                final_response=None,
                metadata={"workflow_type": "langgraph"}
            )
            
            # 执行工作流
            result = self.workflow.invoke(initial_state)
            
            return {
                "response": result.get("final_response", ""),
                "chat_history": chat_history or [],
                "metadata": {
                    "agent_type": "langgraph_multi_agent",
                    "agent_responses": result.get("agent_responses", {}),
                    "current_agent": result.get("current_agent", ""),
                    "workflow_type": "langgraph"
                }
            }
            
        except Exception as e:
            logger.error(f"多智能体系统执行失败: {str(e)}")
            return {
                "response": f"处理失败: {str(e)}",
                "chat_history": chat_history or [],
                "metadata": {
                    "error": str(e),
                    "agent_type": "langgraph_multi_agent"
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

    def _convert_chat_history_for_react(self, chat_history: List[BaseMessage]) -> List[str]:
        """
        转换聊天历史格式为字符串列表，用于 ReAct 智能体
        
        Args:
            chat_history: Langchain 消息格式
            
        Returns:
            字符串列表
        """
        return [message.content for message in chat_history]

def create_langgraph_multi_agent_system(
    model_config: Dict[str, Any],
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False
) -> LangGraphMultiAgentSystem:
    """
    创建 LangGraph 多智能体系统
    
    Args:
        model_config: 模型配置
        tools: 工具列表
        verbose: 是否输出详细日志
        
    Returns:
        LangGraphMultiAgentSystem 实例
    """
    # 创建模型
    model = create_langchain_model(model_config)
    
    # 创建多智能体系统
    system = LangGraphMultiAgentSystem(
        model=model,
        tools=tools,
        verbose=verbose
    )
    
    return system 