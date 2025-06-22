"""
工具工厂
直接创建和返回 Langchain 兼容的工具实例
"""
import os
from typing import List
from langchain.tools import Tool

from src.tools.calculator_tool import CalculatorTool
from src.tools.medical_assessment_tool import MedicalAssessmentTool
from src.tools.medical_reference_tool import MedicalReferenceTool
from src.tools.search_tool import SearchTool, BingSearchTool
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def create_tools() -> List[Tool]:
    """
    创建默认的 Langchain 工具列表
    
    Returns:
        工具列表
    """
    tools = []
    
    # 计算工具
    calculator_tool_instance = CalculatorTool()
    calculator_tool = Tool(
        name="Calculator",
        func=calculator_tool_instance.run,
        description="A calculator tool for performing mathematical calculations."
    )
    tools.append(calculator_tool)
    
    # 医疗评估工具
    medical_assessment_tool_instance = MedicalAssessmentTool()
    medical_assessment_tool = Tool(
        name="Medical Assessment",
        func=medical_assessment_tool_instance.run,
        description="A tool for medical symptom assessment and preliminary diagnosis."
    )
    tools.append(medical_assessment_tool)
    
    # 医疗参考工具
    medical_reference_tool_instance = MedicalReferenceTool()
    medical_reference_tool = Tool(
        name="Medical Reference",
        func=medical_reference_tool_instance.run,
        description="A tool for providing medical reference information and guidelines."
    )
    tools.append(medical_reference_tool)
    
    # 搜索工具（需要配置API密钥）
    search_tool_instance = SearchTool()
    search_tool = Tool(
        name="Web Search",
        func=search_tool_instance.run,
        description="A search engine tool for finding current information on the web."
    )
    tools.append(search_tool)
    
    return tools 