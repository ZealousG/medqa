# SelfMedQA 工具和智能体系统

## 概述

SelfMedQA 是一个基于大语言模型的医疗问答系统，包含多种工具和智能体，能够处理复杂的医疗推理任务。

## 架构设计

### 工具系统 (src/tools/)
- **基础工具**：提供单一功能的工具，如计算、搜索、医疗评估等
- **工具适配器**：将自定义工具适配到不同框架（如LangChain）

### 智能体系统 (src/agents/)
- **单一智能体**：基于LangChain的医疗智能体
- **多智能体系统**：基于LangGraph的多智能体协作系统
- **ReAct智能体**：基于推理-行动模式的智能体，用于复杂任务处理

## ReAct 智能体

### 概述
ReAct (Reasoning and Acting) 智能体是一种能够进行多步推理和行动的智能体架构，特别适合处理复杂的医疗推理任务。

### 特点
- **多步推理**：能够分解复杂问题，进行逐步分析和推理
- **工具调用**：在推理过程中调用各种工具获取信息
- **自主决策**：根据推理结果自主选择下一步行动
- **可配置性**：支持自定义推理步骤数、工具集等

### 使用示例

```python
from src.agents.react_agent import create_react_agent
from src.models.langchain_model_adapter import create_langchain_model

# 创建模型
model_config = {
    "model_type": "qwen",
    "model_name": "Qwen/Qwen2-7B-Instruct",
    "device": "cuda"
}
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
    max_steps=8,
    verbose=True
)

# 执行复杂任务
result = react_agent.run("请分析一个45岁男性患者的心血管疾病风险...")
```

### 在多智能体系统中的集成

ReAct智能体可以集成到多智能体系统中，作为专门的推理智能体：

```python
from src.agents.langgraph_multi_agent import create_langgraph_multi_agent_system

# 创建多智能体系统（自动包含ReAct智能体）
multi_agent_system = create_langgraph_multi_agent_system(
    model_config=model_config,
    verbose=True
)

# 系统会自动路由复杂推理任务给ReAct智能体
result = multi_agent_system.run("请综合分析患者的整体健康状况...")
```

## 工具列表

### 基础工具
所有工具都直接继承LangChain的`BaseTool`类，无需额外的适配层：

1. **计算工具** (`calculator_tool.py`)
   - 数学计算
   - 医疗剂量计算
   - BMI计算
   - 体表面积计算

2. **医疗评估工具** (`medical_assessment_tool.py`)
   - 心血管疾病风险评估
   - 糖尿病风险评估
   - 抑郁症筛查
   - 焦虑症筛查

3. **搜索工具** (`search_tool.py`)
   - 网络搜索
   - 医疗信息检索

4. **医疗参考工具** (`medical_reference_tool.py`)
   - 药物信息查询
   - 疾病信息查询
   - 治疗方案参考

### 工具工厂
- **工具工厂** (`tools.py`)
  - 提供统一的工具创建接口
  - 直接返回LangChain兼容的工具实例

## 智能体列表

### 单一智能体
1. **LangChain医疗智能体** (`langchain_agent.py`)
   - 基于LangChain的医疗问答智能体
   - 支持工具调用和对话历史

2. **智能体工厂** (`langchain_agent_factory.py`)
   - 智能体创建和管理
   - 配置化的智能体构建

### 多智能体系统
1. **LangGraph多智能体系统** (`langgraph_multi_agent.py`)
   - 基于LangGraph的多智能体协作
   - 包含诊断、治疗、预防、推理四个专业智能体
   - 智能任务路由和协调

2. **ReAct智能体** (`react_agent.py`)
   - 专门处理复杂推理任务
   - 多步思考和工具调用
   - 可集成到多智能体系统中

## 使用流程

### 1. 简单任务
对于简单的医疗问答，使用单一智能体：

```python
from src.agents.langchain_agent import create_langchain_medical_agent

agent = create_langchain_medical_agent(model_config)
result = agent.run("什么是高血压？")
```

### 2. 复杂推理任务
对于需要多步推理的复杂任务，使用ReAct智能体：

```python
from src.agents.react_agent import create_react_agent

react_agent = create_react_agent(model, tools)
result = react_agent.run("请分析患者的心血管疾病风险并制定预防方案")
```

### 3. 多专业协作任务
对于需要多个专业领域协作的任务，使用多智能体系统：

```python
from src.agents.langgraph_multi_agent import create_langgraph_multi_agent_system

multi_agent = create_langgraph_multi_agent_system(model_config)
result = multi_agent.run("请为糖尿病患者制定全面的健康管理计划")
```

## 配置说明

### 模型配置
```python
model_config = {
    "model_type": "qwen",  # 模型类型
    "model_name": "Qwen/Qwen2-7B-Instruct",  # 模型名称
    "device": "cuda",  # 设备
    "max_length": 2048,  # 最大长度
    "temperature": 0.7  # 温度参数
}
```

### ReAct智能体配置
```python
react_config = {
    "name": "医疗推理智能体",  # 智能体名称
    "description": "用于处理复杂医疗推理任务",  # 描述
    "max_steps": 8,  # 最大推理步骤
    "verbose": True  # 详细输出
}
```

## 扩展开发

### 添加新工具
1. 继承LangChain的`BaseTool`类
2. 实现`_run`方法
3. 在工具工厂中注册

### 添加新智能体
1. 创建智能体类
2. 实现`run`方法
3. 在多智能体系统中集成

### 自定义ReAct智能体
1. 继承`ReActAgent`类
2. 重写推理逻辑
3. 添加自定义工具

## 注意事项

1. **医疗安全**：所有工具和智能体都提供医疗信息参考，不替代专业医疗诊断
2. **数据隐私**：确保患者数据的隐私保护
3. **模型限制**：注意大语言模型的局限性，重要决策需要人工审核
4. **工具依赖**：确保所有依赖的工具和API正常工作

## 测试覆盖

### 测试文件
- `test/test_tools/` - 工具功能测试
- `test/test_agents/` - 智能体功能测试
- `examples/` - 使用示例