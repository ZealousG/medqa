"""
示例包
包含各种使用示例
"""

__version__ = "1.0.0"
__author__ = "SelfMedQA Team"

# 导入所有示例模块
from . import qa_model_example
from .agents import qa_agent_example
from . import langchain_agent_example
from . import react_agent_example
from . import multi_model_graph_example

__all__ = [
    'qa_model_example',
    'qa_agent_example', 
    'langchain_agent_example',
    'react_agent_example',
    'multi_model_graph_example'
] 