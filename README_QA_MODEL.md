# QA模型系统

QA模型系统提供了灵活的模型管理功能，支持API模式和本地模型模式，统一使用 `ModelConfig` 配置。

## 特性

- **统一配置管理**: 所有模型配置都通过 `src/configs/model_config.py` 中的 `ModelConfig` 类管理
- **多种模型策略**: 支持API模式（OpenAI、火山引擎）和本地模型模式
- **模型缓存**: 自动缓存模型实例，提高性能
- **直接LangChain集成**: 直接返回LangChain的 `BaseLLM` 实例，无需适配器
- **灵活配置**: 支持自定义配置参数

## 配置系统

### ModelConfig 配置类

所有模型配置都通过 `ModelConfig` 类统一管理：

```python
from src.configs.model_config import ModelConfig

# 创建配置实例
config = ModelConfig()

# 配置API模式
config.use_api = True  # 使用火山引擎API，False为OpenAI
config.volc_model_name = "qwen-turbo"
config.volc_api_base = "https://api.volcengine.com"
config.volc_api_key = "your_api_key"

# 配置本地模型
config.model_path = "/path/to/your/model"
config.device = "cuda:0"

# 配置推理参数
config.temperature = 0.7
config.max_length = 2048
```

### 主要配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_api` | bool | False | 是否使用API模式 |
| `model_path` | str | "/mnt/d/data/LLMs/Qwen/Qwen2.5-1.5B" | 本地模型路径 |
| `device` | str | "cuda:0" | 设备类型 |
| `temperature` | float | 0.7 | 采样温度 |
| `max_length` | int | 2048 | 最大输入长度 |
| `volc_model_name` | str | None | 火山引擎模型名称 |
| `volc_api_base` | str | None | 火山引擎API地址 |
| `volc_api_key` | str | None | 火山引擎API密钥 |

## 使用方法

### 1. 基本使用

```python
from src.models.QA_model import QAModel, create_qa_model

# 方法1：使用QAModel类
qa_model = QAModel()
api_model = qa_model.create_api_model()
local_model = qa_model.create_local_model()

# 方法2：使用便捷函数
model = create_qa_model("api")  # 或 "local"
```

### 2. 自定义配置

```python
from src.configs.model_config import ModelConfig
from src.models.QA_model import QAModel

# 创建自定义配置
config = ModelConfig()
config.use_api = True  # 使用火山引擎API
config.temperature = 0.5
config.max_length = 1024

# 使用自定义配置创建模型
qa_model = QAModel(config)
model = qa_model.create_api_model()
```

### 3. 在QA Agent中使用

```python
from src.agents.QA_agent import QA_Agent
from src.configs.model_config import ModelConfig

# 使用默认配置
agent = QA_Agent(verbose=True)

# 使用自定义配置
config = ModelConfig()
config.use_api = True
agent = QA_Agent(model_config=config, verbose=True)

# 运行查询
result = agent.run("什么是高血压？")
print(result['response'])
```

### 4. 模型切换

```python
# 创建Agent
agent = QA_Agent(verbose=True)

# 使用API模型
result1 = agent.run("头痛的原因？")

# 切换到本地模型
agent.switch_model("local")
result2 = agent.run("头痛的原因？")
```

## 模型策略

### APIModelStrategy

API模型策略支持两种模式：

1. **OpenAI API** (`use_api = False`)
   - 使用OpenAI的GPT模型
   - 需要设置 `OPENAI_API_KEY` 环境变量

2. **火山引擎API** (`use_api = True`)
   - 使用火山引擎的Qwen模型
   - 需要设置 `VOLC_API_KEY`、`VOLC_API_BASE`、`VOLC_MODEL` 环境变量

### LocalModelStrategy

本地模型策略：
- 加载本地训练的模型
- 支持GPU加速
- 可配置设备类型和推理参数

## 缓存机制

系统自动缓存模型实例以提高性能：

```python
qa_model = QAModel()

# 第一次创建（会缓存）
model1 = qa_model.create_api_model()

# 第二次创建（使用缓存）
model2 = qa_model.create_api_model()

# 检查是否是同一个实例
print(model1 is model2)  # True

# 清除缓存
qa_model.clear_cache()

# 查看缓存的模型
cached_models = qa_model.get_cached_models()
```

## 错误处理

系统提供完善的错误处理机制：

```python
try:
    # 尝试创建无效模型类型
    model = qa_model.create_model("invalid_type")
except ValueError as e:
    print(f"错误: {e}")

try:
    # 尝试使用无效配置
    config = ModelConfig()
    config.model_path = "/invalid/path"
    qa_model = QAModel(config)
    model = qa_model.create_local_model()
except Exception as e:
    print(f"配置错误: {e}")
```

## 环境变量配置

创建 `.env` 文件配置API密钥：

```bash
# OpenAI API
OPENAI_API_KEY=your_openai_api_key

# 火山引擎API
VOLC_API_KEY=your_volc_api_key
VOLC_API_BASE=https://api.volcengine.com
VOLC_MODEL=qwen-turbo
```

## 示例

### 完整示例

```python
from src.configs.model_config import ModelConfig
from src.models.QA_model import QAModel
from src.agents.QA_agent import QA_Agent

# 1. 配置模型
config = ModelConfig()
config.use_api = True  # 使用火山引擎API
config.temperature = 0.7
config.max_length = 2048

# 2. 创建模型
qa_model = QAModel(config)
model = qa_model.create_api_model()

# 3. 创建Agent
agent = QA_Agent(model=model, verbose=True)

# 4. 运行查询
result = agent.run("什么是高血压？")
print(f"查询: {result['query']}")
print(f"响应: {result['response']}")
print(f"模型类型: {result['model_type']}")

# 5. 查看模型信息
model_info = agent.get_model_info()
print(f"模型信息: {model_info}")
```

### 运行示例

```bash
# 运行QA模型示例
python src/examples/qa_model_example.py

# 运行QA Agent示例
python src/examples/qa_agent_example.py

# 运行测试
python test_qa_model.py
```

## 架构设计

```
src/
├── configs/
│   └── model_config.py          # 统一配置管理
├── models/
│   ├── QA_model.py             # QA模型主类
│   ├── qwen_model.py           # Qwen2本地模型
│   └── base_model.py           # 基础模型类
├── agents/
│   └── QA_agent.py             # QA Agent
└── examples/
    ├── qa_model_example.py     # 模型使用示例
    └── qa_agent_example.py     # Agent使用示例
```

## 优势

1. **统一配置**: 所有配置都通过 `ModelConfig` 管理，避免配置分散
2. **简化使用**: 移除了复杂的适配器层，直接使用LangChain原生模型
3. **灵活切换**: 支持运行时切换不同模型类型
4. **性能优化**: 自动缓存机制提高性能
5. **错误处理**: 完善的错误处理和日志记录

## 注意事项

1. 确保环境变量正确配置
2. 本地模型需要足够的GPU内存
3. API模式需要网络连接
4. 建议在生产环境中使用模型缓存 