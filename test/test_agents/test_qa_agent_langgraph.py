#!/usr/bin/env python3
"""
QA Agent LangGraph 工作流测试
测试API形式的模型进行医疗问答
"""
import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from typing import Dict, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.QA_agent import QA_Agent, AgentState
from src.configs.model_config import ModelConfig
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class TestQAAgentLangGraph(unittest.TestCase):
    """QA Agent LangGraph 测试类"""
    
    def setUp(self):
        """测试前的设置"""
        # 配置API模型
        self.model_config = ModelConfig(
            use_api=True,
            api_key="test-api-key",
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=1000
        )
        
        # 创建QA Agent
        self.qa_agent = QA_Agent(
            model_type="api",
            model_config=self.model_config,
            verbose=False
        )
    
    def test_agent_initialization(self):
        """测试Agent初始化"""
        self.assertIsNotNone(self.qa_agent)
        self.assertEqual(self.qa_agent.model_config.use_api, True)
        self.assertEqual(self.qa_agent.model_config.model, "gpt-3.5-turbo")
        self.assertTrue(len(self.qa_agent.tools) > 0)
    
    def test_workflow_creation(self):
        """测试工作流创建"""
        workflow = self.qa_agent.workflow
        self.assertIsNotNone(workflow)
        
        # 检查工作流节点
        model_info = self.qa_agent.get_model_info()
        expected_nodes = ["classification_agent", "diagnosis_agent", "treatment_agent", "prevention_agent", "summary"]
        self.assertEqual(model_info["workflow_nodes"], expected_nodes)
    
    def test_get_available_tools(self):
        """测试获取可用工具"""
        tools = self.qa_agent.get_available_tools()
        self.assertIsInstance(tools, list)
        self.assertTrue(len(tools) > 0)
        
        # 检查是否包含预期的工具
        expected_tools = ["Calculator", "Medical Assessment", "Medical Reference", "Web Search"]
        for tool in expected_tools:
            self.assertIn(tool, tools)
    
    def test_get_model_info(self):
        """测试获取模型信息"""
        model_info = self.qa_agent.get_model_info()
        
        self.assertIn("model_type", model_info)
        self.assertIn("model_name", model_info)
        self.assertIn("tools_count", model_info)
        self.assertIn("tools", model_info)
        self.assertIn("workflow_nodes", model_info)
        self.assertIn("config", model_info)
        
        self.assertIsInstance(model_info["tools_count"], int)
        self.assertIsInstance(model_info["tools"], list)
        self.assertIsInstance(model_info["workflow_nodes"], list)
    
    @patch('src.agents.QA_agent.create_qa_model')
    def test_switch_model(self, mock_create_model):
        """测试切换模型"""
        mock_model = MagicMock()
        mock_create_model.return_value = mock_model
        
        self.qa_agent.switch_model("gpt-4")
        
        mock_create_model.assert_called_once_with("gpt-4", self.model_config)
        self.assertEqual(self.qa_agent.model, mock_model)
    
    def test_convert_chat_history(self):
        """测试聊天历史转换"""
        chat_history = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "您好，有什么可以帮助您的吗？"},
            {"role": "user", "content": "我头痛"}
        ]
        
        messages = self.qa_agent._convert_chat_history(chat_history)
        
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[0].content, "你好")
        self.assertEqual(messages[1].content, "您好，有什么可以帮助您的吗？")
        self.assertEqual(messages[2].content, "我头痛")
    
    def test_route_to_agents(self):
        """测试路由到Agent"""
        # 测试诊断Agent路由
        state = AgentState(
            messages=[],
            current_agent="diagnosis_agent",
            agent_responses={},
            final_response=None,
            metadata={}
        )
        result = self.qa_agent._route_to_agents(state)
        self.assertEqual(result, "diagnosis_agent")
        
        # 测试治疗Agent路由
        state["current_agent"] = "treatment_agent"
        result = self.qa_agent._route_to_agents(state)
        self.assertEqual(result, "treatment_agent")
        
        # 测试预防Agent路由
        state["current_agent"] = "prevention_agent"
        result = self.qa_agent._route_to_agents(state)
        self.assertEqual(result, "prevention_agent")
        
        # 测试默认路由
        state["current_agent"] = "unknown"
        result = self.qa_agent._route_to_agents(state)
        self.assertEqual(result, "diagnosis_agent")
    
    @patch('src.agents.QA_agent.ChatPromptTemplate')
    @patch('src.agents.QA_agent.create_openai_tools_agent')
    @patch('src.agents.QA_agent.AgentExecutor')
    def test_diagnosis_agent_node(self, mock_executor, mock_create_agent, mock_prompt):
        """测试诊断Agent节点"""
        # 模拟Agent执行器
        mock_executor_instance = MagicMock()
        mock_executor.return_value = mock_executor_instance
        mock_executor_instance.invoke.return_value = {"output": "这是一个诊断建议"}
        
        # 创建测试状态
        state = AgentState(
            messages=[MagicMock(content="我头痛")],
            current_agent="diagnosis_agent",
            agent_responses={},
            final_response=None,
            metadata={}
        )
        
        # 执行诊断Agent节点
        result = self.qa_agent._diagnosis_agent_node(state)
        
        # 验证结果
        self.assertIn("诊断Agent", result["agent_responses"])
        self.assertEqual(result["agent_responses"]["诊断Agent"], "这是一个诊断建议")
    
    @patch('src.agents.QA_agent.ChatPromptTemplate')
    @patch('src.agents.QA_agent.create_openai_tools_agent')
    @patch('src.agents.QA_agent.AgentExecutor')
    def test_treatment_agent_node(self, mock_executor, mock_create_agent, mock_prompt):
        """测试治疗Agent节点"""
        # 模拟Agent执行器
        mock_executor_instance = MagicMock()
        mock_executor.return_value = mock_executor_instance
        mock_executor_instance.invoke.return_value = {"output": "这是一个治疗建议"}
        
        # 创建测试状态
        state = AgentState(
            messages=[MagicMock(content="我需要吃什么药")],
            current_agent="treatment_agent",
            agent_responses={},
            final_response=None,
            metadata={}
        )
        
        # 执行治疗Agent节点
        result = self.qa_agent._treatment_agent_node(state)
        
        # 验证结果
        self.assertIn("治疗Agent", result["agent_responses"])
        self.assertEqual(result["agent_responses"]["治疗Agent"], "这是一个治疗建议")
    
    @patch('src.agents.QA_agent.ChatPromptTemplate')
    @patch('src.agents.QA_agent.create_openai_tools_agent')
    @patch('src.agents.QA_agent.AgentExecutor')
    def test_prevention_agent_node(self, mock_executor, mock_create_agent, mock_prompt):
        """测试预防Agent节点"""
        # 模拟Agent执行器
        mock_executor_instance = MagicMock()
        mock_executor.return_value = mock_executor_instance
        mock_executor_instance.invoke.return_value = {"output": "这是一个预防建议"}
        
        # 创建测试状态
        state = AgentState(
            messages=[MagicMock(content="如何预防疾病")],
            current_agent="prevention_agent",
            agent_responses={},
            final_response=None,
            metadata={}
        )
        
        # 执行预防Agent节点
        result = self.qa_agent._prevention_agent_node(state)
        
        # 验证结果
        self.assertIn("预防Agent", result["agent_responses"])
        self.assertEqual(result["agent_responses"]["预防Agent"], "这是一个预防建议")
    
    @patch('src.agents.QA_agent.ChatPromptTemplate')
    def test_summary_node(self, mock_prompt):
        """测试总结节点"""
        # 模拟模型响应
        mock_chain = MagicMock()
        mock_prompt.from_messages.return_value = mock_prompt
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)
        mock_chain.invoke.return_value = MagicMock(content="这是总结后的回答")
        
        # 创建测试状态
        state = AgentState(
            messages=[MagicMock(content="测试查询")],
            current_agent="diagnosis_agent",
            agent_responses={"诊断Agent": "原始诊断回答"},
            final_response=None,
            metadata={}
        )
        
        # 执行总结节点
        result = self.qa_agent._summary_node(state)
        
        # 验证结果
        self.assertIsNotNone(result["final_response"])
    
    @patch('src.agents.QA_agent.ChatPromptTemplate')
    def test_classification_agent_node(self, mock_prompt):
        """测试分类Agent节点"""
        # 模拟模型响应
        mock_chain = MagicMock()
        mock_prompt.from_messages.return_value = mock_prompt
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)
        mock_chain.invoke.return_value = MagicMock(content="诊断Agent")
        
        # 创建测试状态
        state = AgentState(
            messages=[MagicMock(content="我头痛")],
            current_agent="",
            agent_responses={},
            final_response=None,
            metadata={}
        )
        
        # 执行分类Agent节点
        result = self.qa_agent._classification_agent_node(state)
        
        # 验证结果
        self.assertEqual(result["current_agent"], "diagnosis_agent")

class TestQAAgentIntegration(unittest.TestCase):
    """QA Agent 集成测试类"""
    
    def setUp(self):
        """测试前的设置"""
        self.model_config = ModelConfig(
            use_api=True,
            api_key="test-api-key",
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=1000
        )
    
    @patch('src.agents.QA_agent.create_qa_model')
    def test_agent_creation_with_api(self, mock_create_model):
        """测试使用API创建Agent"""
        mock_model = MagicMock()
        mock_create_model.return_value = mock_model
        
        qa_agent = QA_Agent(
            model_type="api",
            model_config=self.model_config,
            verbose=False
        )
        
        self.assertIsNotNone(qa_agent)
        mock_create_model.assert_called_once_with("api", self.model_config)
    
    def test_agent_creation_with_model_instance(self):
        """测试使用模型实例创建Agent"""
        mock_model = MagicMock()
        
        qa_agent = QA_Agent(
            model=mock_model,
            model_config=self.model_config,
            verbose=False
        )
        
        self.assertIsNotNone(qa_agent)
        self.assertEqual(qa_agent.model, mock_model)

if __name__ == "__main__":
    # 运行测试
    unittest.main(verbosity=2) 