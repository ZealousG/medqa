from typing import Any, Dict, List, Optional, TypedDict, Annotated, Union, Generator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.language_models import BaseLLM
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END

from src.models.QA_model import create_qa_model, QAModel
from src.configs.model_config import ModelConfig
from langchain_core.tools import BaseTool
from src.tools.tools import create_tools
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class AgentState(TypedDict):
    """Agent 状态定义"""
    messages: Annotated[List[BaseMessage], "聊天消息列表"]
    current_agent: Annotated[str, "当前执行的 Agent"]
    agent_responses: Annotated[Dict[str, str], "各 Agent 的响应"]
    final_response: Annotated[Optional[str], "最终响应"]
    metadata: Annotated[Dict[str, Any], "元数据"]

class QA_Agent:
    """
    医疗问答 Agent 类
    基于专业医疗Agent设计，支持诊断、治疗、预防等专业领域
    统一使用ModelConfig配置
    """
    
    def __init__(self, 
                 model: Union[BaseLLM, str] = None,
                 model_type: str = "api",
                 model_config: Optional[ModelConfig] = None,
                 tools: Optional[List[BaseTool]] = None, 
                 verbose: bool = False):
        """
        初始化医疗问答 Agent
        
        Args:
            model: 模型实例或模型类型字符串
            model_type: 模型类型 ("api", "local")
            model_config: 模型配置
            tools: 工具列表
            verbose: 是否详细输出
        """
        self.verbose = verbose
        self.model_config = model_config or ModelConfig()
        
        # 处理模型参数
        if isinstance(model, BaseLLM):
            # 直接使用提供的模型实例
            self.model = model
        else:
            # 使用model_type参数创建模型（model可能是字符串或None）
            model_type_to_use = model if isinstance(model, str) else model_type
            self.model = create_qa_model(model_type_to_use, self.model_config)
        
        # 设置工具
        self.tools = tools or create_tools()
        
        # 创建工作流图
        self.workflow = self._create_workflow()
        
        logger.info(f"医疗问答 Agent 初始化完成，模型类型: {type(self.model).__name__}")

    def run(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        运行医疗问答 Agent
        
        Args:
            query: 用户查询
            chat_history: 聊天历史
            
        Returns:
            响应结果
        """
        try:
            if self.verbose:
                logger.info(f"处理医疗查询: {query}")
            
            # 转换聊天历史
            messages = self._convert_chat_history(chat_history) if chat_history else []
            messages.append(HumanMessage(content=query))
            
            # 初始化状态
            initial_state = AgentState(
                messages=messages,
                current_agent="",
                agent_responses={},
                final_response=None,
                metadata={"workflow_type": "qa_agent"}
            )
            
            # 执行工作流
            result = self.workflow.invoke(initial_state)
            
            return {
                "query": query,
                "response": result.get("final_response", ""),
                "agent_type": result.get("current_agent", ""),
                "model_type": type(self.model).__name__,
                "tools_used": self.get_available_tools(),
                "metadata": {
                    "chat_history_length": len(chat_history) if chat_history else 0,
                    "agent_responses": result.get("agent_responses", {}),
                    "workflow_type": "qa_agent"
                }
            }
            
        except Exception as e:
            logger.error(f"医疗问答 Agent 运行失败: {str(e)}")
            return {
                "query": query,
                "response": f"抱歉，处理您的医疗查询时出现错误: {str(e)}",
                "error": str(e),
                "model_type": type(self.model).__name__ if self.model else "Unknown"
            }
    
    def run_stream(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Generator[Dict[str, Any], None, None]:
        """
        流式运行医疗问答 Agent
        
        Args:
            query: 用户查询
            chat_history: 聊天历史
            
        Yields:
            流式响应结果
        """
        try:
            if self.verbose:
                logger.info(f"流式处理医疗查询: {query}")
            
            # 初始响应
            yield {
                "query": query,
                "response": "正在分析您的问题...",
                "status": "processing",
                "stage": "classification"
            }
            
            # 转换聊天历史
            messages = self._convert_chat_history(chat_history) if chat_history else []
            messages.append(HumanMessage(content=query))
            
            # 初始化状态
            initial_state = AgentState(
                messages=messages,
                current_agent="",
                agent_responses={},
                final_response=None,
                metadata={"workflow_type": "qa_agent"}
            )
            
            # 执行分类
            yield {
                "query": query,
                "response": "正在分类您的问题类型...",
                "status": "processing",
                "stage": "classification"
            }
            
            # 执行工作流
            result = self.workflow.invoke(initial_state)
            
            # 最终响应
            final_response = {
                "query": query,
                "response": result.get("final_response", ""),
                "agent_type": result.get("current_agent", ""),
                "model_type": type(self.model).__name__,
                "tools_used": self.get_available_tools(),
                "status": "completed",
                "metadata": {
                    "chat_history_length": len(chat_history) if chat_history else 0,
                    "agent_responses": result.get("agent_responses", {}),
                    "workflow_type": "qa_agent"
                }
            }
            
            yield final_response
            
        except Exception as e:
            logger.error(f"流式医疗问答 Agent 运行失败: {str(e)}")
            yield {
                "query": query,
                "response": f"抱歉，处理您的医疗查询时出现错误: {str(e)}",
                "error": str(e),
                "status": "error",
                "model_type": type(self.model).__name__ if self.model else "Unknown"
            }
    
    def _create_workflow(self) -> StateGraph:
        """
        创建工作流图
        
        Returns:
            StateGraph 实例
        """
        # 创建状态图
        workflow = StateGraph(AgentState)
        
        # 添加节点
        workflow.add_node("classification_agent", self._classification_agent_node)
        workflow.add_node("diagnosis_agent", self._diagnosis_agent_node)
        workflow.add_node("treatment_agent", self._treatment_agent_node)
        workflow.add_node("prevention_agent", self._prevention_agent_node)
        workflow.add_node("summary", self._summary_node)
        
        # 设置入口点
        workflow.set_entry_point("classification_agent")
        
        # 添加条件边
        workflow.add_conditional_edges(
            "classification_agent",
            self._route_to_agents,
            {
                "diagnosis_agent": "diagnosis_agent",
                "treatment_agent": "treatment_agent",
                "prevention_agent": "prevention_agent"
            }
        )
        
        # 添加边到协调器
        workflow.add_edge("diagnosis_agent", "summary")
        workflow.add_edge("treatment_agent", "summary")
        workflow.add_edge("prevention_agent", "summary")
        
        # 设置结束点
        workflow.add_edge("summary", END)
        
        return workflow.compile()
    
    def _classification_agent_node(self, state: AgentState) -> AgentState:
        """
        分类Agent节点
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态
        """
        try:
            # 获取用户查询
            user_message = state["messages"][-1].content if state["messages"] else ""
            
            # 创建分类提示
            classification_prompt = (
                "你是一个医疗查询路由专家，负责根据用户问题自动选择最合适的专业Agent进行处理。请按照以下规则进行分类：\n\n"
                "1. **诊断Agent**：当问题涉及症状描述、疾病识别或身体不适时选择。关键词包括但不限于：\n"
                "   - 症状、诊断、疾病、疼痛、不适\n"
                "   - 具体症状：发烧、咳嗽、头痛、腹痛、恶心、呕吐等\n\n"
                "2. **治疗Agent**：当问题涉及疾病治疗、用药方案或医疗干预时选择。关键词包括但不限于：\n"
                "   - 治疗、药物、用药、手术、康复\n"
                "   - 疗程、剂量、副作用、疗效\n\n"
                "3. **预防Agent**：当问题涉及健康维护或疾病预防时选择。关键词包括但不限于：\n"
                "   - 预防、保健、生活方式、饮食、运动\n"
                "   - 体检、筛查、疫苗、免疫\n\n"
                "**决策流程**：\n"
                "1. 将用户问题转为小写分析\n"
                "2. 优先匹配以下关键词进行判断：\n"
                "   - 出现诊断关键词 → 诊断Agent\n"
                "   - 出现治疗关键词 → 治疗Agent\n"
                "   - 出现预防关键词 → 预防Agent\n"
                "3. 当同时匹配多个类别时：\n"
                "   - 治疗关键词优先于诊断关键词\n"
                "   - 预防关键词优先于诊断关键词\n"
                "4. 无匹配关键词时 → 默认诊断Agent\n\n"
                "请只返回Agent类型（\"诊断Agent\"、\"治疗Agent\"或\"预防Agent\"），不要返回任何其他内容。\n\n"
                f"当前用户问题：\"{user_message}\""
            )
            
            # 使用模型进行分类
            from langchain_core.prompts import ChatPromptTemplate
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "你是一个医疗查询分类专家。"),
                ("human", classification_prompt)
            ])
            
            chain = prompt | self.model
            result = chain.invoke({})
            
            # 解析分类结果
            agent_type = result.content.strip()
            if "治疗" in agent_type:
                state["current_agent"] = "treatment_agent"
            elif "预防" in agent_type:
                state["current_agent"] = "prevention_agent"
            else:
                state["current_agent"] = "diagnosis_agent"  # 默认
            
            if self.verbose:
                logger.info(f"分类结果: {agent_type} -> {state['current_agent']}")
            
        except Exception as e:
            logger.error(f"分类Agent执行失败: {str(e)}")
            state["current_agent"] = "diagnosis_agent"  # 默认
        
        return state
    
    def _diagnosis_agent_node(self, state: AgentState) -> AgentState:
        """
        诊断Agent节点
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态
        """
        try:
            # 诊断Agent提示词
            diagnosis_prompt = (
                "你是一个专注于医疗诊断的Agent。你的职责是分析用户描述的症状和体征，提供可能的诊断和鉴别诊断。"
                "你应该提出关键问题来完善诊断，并解释不同诊断的可能性和依据。"
                "在回答时要保持客观谨慎，避免确定性诊断，而是提供可能性分析和下一步建议。"
            )
            
            # 创建临时 Agent 执行器
            prompt = ChatPromptTemplate.from_messages([
                ("system", diagnosis_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
            
            agent = create_openai_tools_agent(
                llm=self.model,
                tools=self.tools,
                prompt=prompt
            )
            
            agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
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
            state["agent_responses"]["诊断Agent"] = result.get("output", "")
            
            if self.verbose:
                logger.info("诊断Agent 响应: {result.get('output', '')}")
            
        except Exception as e:
            logger.error("诊断Agent 执行失败: {str(e)}")
            state["agent_responses"]["诊断Agent"] = f"执行失败: {str(e)}"
        
        return state
    
    def _treatment_agent_node(self, state: AgentState) -> AgentState:
        """
        治疗Agent节点
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态
        """
        try:
            # 治疗Agent提示词
            treatment_prompt = (
                "你是一个专注于医疗治疗的Agent。你的职责是提供关于治疗方法、药物使用和治疗计划的信息。"
                "你应该基于用户的情况，提供关于常规治疗方案、药物选择、可能的副作用和治疗效果的信息。"
                "在回答时要平衡治疗的效果和风险，提供循证医学的证据支持，同时考虑治疗的个体化。"
            )
            
            # 创建临时 Agent 执行器
            prompt = ChatPromptTemplate.from_messages([
                ("system", treatment_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
            
            agent = create_openai_tools_agent(
                llm=self.model,
                tools=self.tools,
                prompt=prompt
            )
            
            agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
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
            state["agent_responses"]["治疗Agent"] = result.get("output", "")
            
            if self.verbose:
                logger.info("治疗Agent 响应: {result.get('output', '')}")
            
        except Exception as e:
            logger.error("治疗Agent 执行失败: {str(e)}")
            state["agent_responses"]["治疗Agent"] = f"执行失败: {str(e)}"
        
        return state
    
    def _prevention_agent_node(self, state: AgentState) -> AgentState:
        """
        预防Agent节点
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态
        """
        try:
            # 预防Agent提示词
            prevention_prompt = (
                "你是一个专注于疾病预防和健康维护的Agent。你的职责是提供关于健康生活方式、疾病预防和健康监测的信息。"
                "你应该基于用户的情况，提供关于饮食、运动、生活习惯和预防措施的建议，以减少疾病风险和提高生活质量。"
                "在回答时要强调预防的重要性，提供实用的健康建议，同时考虑用户的实际情况和可行性。"
            )
            
            # 创建临时 Agent 执行器
            prompt = ChatPromptTemplate.from_messages([
                ("system", prevention_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
            
            agent = create_openai_tools_agent(
                llm=self.model,
                tools=self.tools,
                prompt=prompt
            )
            
            agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
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
            state["agent_responses"]["预防Agent"] = result.get("output", "")
            
            if self.verbose:
                logger.info("预防Agent 响应: {result.get('output', '')}")
            
        except Exception as e:
            logger.error("预防Agent 执行失败: {str(e)}")
            state["agent_responses"]["预防Agent"] = f"执行失败: {str(e)}"
        
        return state
    
    def _summary_node(self, state: AgentState) -> AgentState:
        """
        总结节点
        
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
                state["final_response"] = response
                return state
            
            # 多个响应时，创建协调提示
            coordination_prompt = "请整合以下专业Agent的回答，提供一个连贯、一致的最终回答：\n\n"
            
            for agent_name, response in responses.items():
                coordination_prompt += f"【{agent_name}】:\n{response}\n\n"
            
            coordination_prompt += "请整合以上回答，提供最终一致的响应。"
            
            # 使用模型进行协调
            from langchain_core.prompts import ChatPromptTemplate
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "你是一个医疗多Agent系统的协调员。你的职责是综合专业Agent的回答，提供最终的一致性响应。"),
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
        else:
            return "diagnosis_agent"  # 默认
    
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
    
    def get_available_tools(self) -> List[str]:
        """获取可用工具列表"""
        return [tool.name for tool in self.tools]
    
    def switch_model(self, model_type: str) -> None:
        """
        切换模型
        
        Args:
            model_type: 模型类型
        """
        self.model = create_qa_model(model_type, self.model_config)
        logger.info(f"模型已切换到: {model_type}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model_type": type(self.model).__name__,
            "model_name": getattr(self.model, 'model_name', getattr(self.model, 'model', 'Unknown')),
            "tools_count": len(self.tools),
            "tools": self.get_available_tools(),
            "workflow_nodes": ["classification_agent", "diagnosis_agent", "treatment_agent", "prevention_agent", "summary"],
            "config": {
                "use_api": self.model_config.use_api,
                "model_path": self.model_config.model_path,
                "device": self.model_config.device,
                "temperature": self.model_config.temperature
            }
        }