# 医疗问答系统 UI 模块

## 概述

本模块提供了基于 Gradio 的医疗问答系统用户界面，支持左侧历史记录显示和右侧对话交互。

## 功能特性

### 🎨 界面设计
- **左侧面板**: 历史记录列表，支持查看、导入、导出
- **右侧面板**: 实时对话区域，支持多轮对话
- **响应式布局**: 适配不同屏幕尺寸
- **现代化UI**: 使用 Gradio Soft 主题

### 💬 对话功能
- **实时对话**: 支持与医疗问答 Agent 实时交互
- **多轮对话**: 保持对话上下文连续性
- **消息显示**: 用户消息和助手回复清晰区分
- **错误处理**: 友好的错误提示和处理

### 📚 历史记录
- **自动保存**: 对话自动保存到本地文件
- **历史查看**: 点击历史记录可重新查看对话
- **导入导出**: 支持历史记录的导入和导出
- **清空功能**: 支持清空当前对话或所有历史

### 🔧 模型管理
- **模型切换**: 支持 API 和本地模型切换
- **配置显示**: 实时显示当前模型配置信息
- **参数调整**: 支持模型参数的自定义配置

## 快速开始

### 1. 安装依赖

```bash
pip install gradio
```

### 2. 基本使用

```python
from src.UI.gradio_interface import GradioInterface

# 创建界面实例
interface = GradioInterface(
    model_type="api",
    verbose=True
)

# 启动界面
interface.launch(
    server_port=7860,
    share=False
)
```

### 3. 自定义配置

```python
from src.UI.gradio_interface import GradioInterface
from src.configs.model_config import ModelConfig

# 创建自定义配置
model_config = ModelConfig(
    use_api=True,
    temperature=0.8,
    max_tokens=1024
)

# 创建界面实例
interface = GradioInterface(
    model_type="api",
    model_config=model_config,
    verbose=True
)

# 启动界面
interface.launch(server_port=7860)
```

## 运行方式

### 方式一：直接运行启动脚本

```bash
cd src/UI
python run_interface.py
```

### 方式二：运行示例

```bash
cd examples/UI_examples
python gradio_example.py
```

### 方式三：在代码中使用

```python
from src.UI.gradio_interface import GradioInterface

interface = GradioInterface()
interface.launch()
```

## 界面说明

### 左侧历史记录面板

- **历史对话列表**: 显示所有历史对话，格式为 "时间 - 问题摘要"
- **清空历史**: 删除所有历史记录
- **导出历史**: 将历史记录保存为 JSON 文件
- **导入历史**: 从 JSON 文件导入历史记录
- **模型信息**: 显示当前模型配置信息

### 右侧对话面板

- **对话显示区域**: 显示当前对话内容
- **输入框**: 输入医疗问题或症状描述
- **发送按钮**: 发送消息给医疗助手
- **清空对话**: 清空当前对话显示
- **模型切换**: 切换 API 或本地模型

## 配置选项

### GradioInterface 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_type` | str | "api" | 模型类型 ("api" 或 "local") |
| `model_config` | ModelConfig | None | 模型配置对象 |
| `verbose` | bool | False | 是否显示详细日志 |

### launch 方法参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `server_name` | str | "0.0.0.0" | 服务器地址 |
| `server_port` | int | 7860 | 服务器端口 |
| `share` | bool | False | 是否生成公共链接 |
| `debug` | bool | True | 是否开启调试模式 |

## 文件结构

```
src/UI/
├── __init__.py              # 包初始化文件
├── gradio_interface.py      # 主要界面类
├── run_interface.py         # 启动脚本
└── README.md               # 说明文档
```

## 历史记录格式

历史记录以 JSON 格式保存，包含以下字段：

```json
{
  "id": 1,
  "timestamp": "2024-01-01T12:00:00",
  "user_input": "我最近头痛",
  "assistant_response": "头痛可能的原因包括...",
  "agent_type": "diagnosis_agent",
  "model_type": "Qwen2_5_model",
  "tools_used": ["medical_reference_tool"],
  "metadata": {
    "chat_history_length": 0,
    "workflow_type": "qa_agent"
  }
}
```

## 注意事项

1. **依赖要求**: 确保已安装 `gradio` 包
2. **模型配置**: 使用 API 模型时需要正确配置 API 密钥
3. **端口冲突**: 如果端口被占用，可以修改 `server_port` 参数
4. **历史文件**: 历史记录默认保存为 `chat_history.json`
5. **浏览器兼容**: 建议使用现代浏览器访问界面

## 故障排除

### 常见问题

1. **界面无法启动**
   - 检查端口是否被占用
   - 确认依赖包已正确安装

2. **模型连接失败**
   - 检查 API 密钥配置
   - 确认网络连接正常

3. **历史记录无法保存**
   - 检查文件写入权限
   - 确认磁盘空间充足

### 日志查看

启用 `verbose=True` 可以查看详细的运行日志，帮助诊断问题。

## 扩展开发

如需扩展界面功能，可以：

1. 继承 `GradioInterface` 类
2. 重写相关方法
3. 添加新的界面组件
4. 自定义事件处理逻辑

## 许可证

本项目采用 MIT 许可证。 