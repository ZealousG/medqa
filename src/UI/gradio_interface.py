import gradio as gr
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Generator
import sys
import os
import asyncio
import markdown
import uuid

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.agents.QA_agent import QA_Agent
from src.configs.model_config import ModelConfig
from src.utils.logger import setup_logger
from langchain_openai import ChatOpenAI

logger = setup_logger("gradio_interface", log_dir="logs")

class GradioInterface:
    """
    Gradio 医疗问答界面
    左侧显示历史记录，右侧显示对话
    """
    
    def __init__(self, 
                 model_type: str = "api",
                 model_config: Optional[ModelConfig] = None,
                 verbose: bool = False):
        """
        初始化 Gradio 界面
        
        Args:
            model_type: 模型类型
            model_config: 模型配置
            verbose: 是否详细输出
        """
        self.model_type = model_type
        self.model_config = model_config or ModelConfig()
        self.verbose = verbose
        
        # 初始化 QA Agent
        self.qa_agent = QA_Agent(
            model_type=model_type,
            model_config=model_config,
            verbose=verbose
        )
        
        # 初始化话题生成模型
        self._init_topic_model()
        
        # 历史记录存储 - 改为按话题存储
        self.topics: List[Dict[str, Any]] = []  # 话题列表
        self.current_topic_id: Optional[str] = None  # 当前话题ID
        self.history_file = "chat_history.json"
        
        # 清除旧格式记录，重新开始
        self._clear_old_history()
        self._load_history()
        
        # 流式输出相关
        self.streaming_response = ""
        self.current_streaming_topic_id = None
        
        # 创建界面
        self.interface = self._create_interface()
    
    def _init_topic_model(self):
        """初始化话题生成模型"""
        try:
            self.topic_model = ChatOpenAI(
                api_key=self.model_config.volc_api_key,
                base_url=self.model_config.volc_api_base,
                model=self.model_config.volc_model_name,
                temperature=0.3,
                max_tokens=50
            )
        except Exception as e:
            logger.error(f"话题生成模型初始化失败: {str(e)}")
            self.topic_model = None
    
    def _generate_topic_title(self, first_question: str) -> str:
        """根据第一个问题生成话题标题"""
        if not self.topic_model:
            return first_question[:30] + "..." if len(first_question) > 30 else first_question
        
        try:
            prompt = f"""请根据以下医疗问题生成一个简洁的话题标题（不超过15个字）：

问题：{first_question}

要求：
1. 标题要概括问题的核心内容
2. 使用医疗相关的简洁词汇
3. 不超过15个字
4. 只返回标题，不要其他内容

话题标题："""
            
            response = self.topic_model.invoke(prompt)
            title = response.content.strip()
            
            # 如果生成失败或太长，使用默认方式
            if len(title) > 20 or not title:
                return first_question[:15] + "..." if len(first_question) > 15 else first_question
            
            return title
            
        except Exception as e:
            logger.error(f"生成话题标题失败: {str(e)}")
            return first_question[:15] + "..." if len(first_question) > 15 else first_question
    
    def _create_interface(self) -> gr.Blocks:
        """创建 Gradio 界面"""
        
        with gr.Blocks(
            title="医疗问答系统",
            theme=gr.themes.Soft(),
            css="""
            .chat-container {
                height: 600px;
                overflow-y: auto;
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 10px;
                background-color: #f9f9f9;
            }
            .history-container {
                height: 500px;
                overflow-y: auto;
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 10px;
                background-color: #f5f5f5;
            }
            .message {
                margin: 10px 0;
                padding: 15px;
                border-radius: 8px;
                max-width: 85%;
                line-height: 1.6;
            }
            .user-message {
                background-color: #007bff;
                color: white;
                margin-left: auto;
                text-align: right;
            }
            .assistant-message {
                background-color: #007bff;
                color: black;
                margin-right: auto;
                border: 1px solid #e0e0e0;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            .assistant-message h1, .assistant-message h2, .assistant-message h3 {
                color: #2c3e50;
                margin-top: 20px;
                margin-bottom: 10px;
            }
            .assistant-message h1 { font-size: 1.5em; }
            .assistant-message h2 { font-size: 1.3em; }
            .assistant-message h3 { font-size: 1.1em; }
            .assistant-message ul, .assistant-message ol {
                margin-left: 20px;
                margin-bottom: 10px;
            }
            .assistant-message li {
                margin-bottom: 5px;
            }
            .assistant-message p {
                margin-bottom: 10px;
            }
            .assistant-message code {
                background-color: #f8f9fa;
                padding: 2px 4px;
                border-radius: 3px;
                font-family: monospace;
            }
            .assistant-message pre {
                background-color: #f8f9fa;
                padding: 10px;
                border-radius: 5px;
                overflow-x: auto;
            }
            .assistant-message blockquote {
                border-left: 4px solid #007bff;
                margin-left: 0;
                padding-left: 15px;
                color: #666;
            }
            .assistant-message strong {
                color: #2c3e50;
            }
            .streaming-indicator {
                display: inline-block;
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background-color: #007bff;
                animation: pulse 1.5s ease-in-out infinite;
            }
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }
            .topic-item {
                padding: 10px;
                margin: 5px 0;
                border-radius: 8px;
                cursor: pointer;
                border: 1px solid #0056b3;
                background-color: #007bff;
                color: white;
                transition: background-color 0.3s;
                user-select: none;
            }
            .topic-item:hover {
                background-color: #0056b3;
            }
            .topic-item.active {
                background-color: #0056b3;
                color: white;
                border: 2px solid #ffffff;
            }
            .topic-title {
                font-weight: bold;
                font-size: 14px;
                margin-bottom: 4px;
            }
            .topic-info {
                font-size: 11px;
                color: #e0e0e0;
            }
            .topic-item.active .topic-info {
                color: #f0f0f0;
            }
            """
        ) as interface:
            
            gr.Markdown("# 🏥 医疗问答系统")
            gr.Markdown("### 专业的医疗咨询助手，为您提供诊断、治疗和预防建议")
            
            with gr.Row():
                # 左侧历史记录面板
                with gr.Column(scale=1):
                    gr.Markdown("### 📚 话题历史")
                    
                    # 话题列表显示
                    topics_display = gr.HTML(
                        value=self._render_topics(),
                        label="历史话题",
                        elem_classes=["history-container"]
                    )
                    
                    # 话题选择下拉菜单
                    topic_selector = gr.Dropdown(
                        choices=self._get_topic_choices(),
                        label="选择话题查看历史",
                        value=None,
                        interactive=True
                    )
                    
                    # 查看话题按钮
                    load_topic_btn = gr.Button("📖 查看选中话题", variant="secondary", size="sm")
                    
                    # 历史记录操作按钮
                    with gr.Row():
                        new_topic_btn = gr.Button("➕ 新话题", variant="primary", size="sm")
                        clear_history_btn = gr.Button("🗑️ 清空历史", variant="secondary", size="sm")
                    
                    with gr.Row():
                        export_history_btn = gr.Button("📤 导出历史", variant="secondary", size="sm")
                        import_history_btn = gr.Button("📥 导入历史", variant="secondary", size="sm")
                        refresh_topics_btn = gr.Button("🔄 刷新话题", variant="secondary", size="sm")
                
                # 右侧对话面板
                with gr.Column(scale=2):
                    gr.Markdown("### 💬 对话区域")
                    
                    # 当前话题标题
                    current_topic_title = gr.Markdown("**当前话题：** 新对话")
                    
                    # 对话显示区域
                    chat_display = gr.HTML(
                        value="<div class='chat-container'><p style='text-align: center; color: #666;'>开始您的医疗咨询...</p></div>",
                        label="对话内容",
                        elem_classes=["chat-container"]
                    )
                    
                    # 输入区域
                    with gr.Row():
                        user_input = gr.Textbox(
                            placeholder="请描述您的症状或医疗问题...",
                            label="输入问题",
                            lines=3,
                            max_lines=5
                        )
                    
                    # 操作按钮
                    with gr.Row():
                        send_btn = gr.Button("🚀 发送", variant="primary")
                        clear_chat_btn = gr.Button("🗑️ 清空对话", variant="secondary")
                    
                    # 模型切换
                    with gr.Row():
                        model_selector = gr.Dropdown(
                            choices=["api", "local"],
                            value=self.model_type,
                            label="模型类型",
                            interactive=True
                        )
                        switch_model_btn = gr.Button("🔄 切换模型", variant="secondary")
            
            # 隐藏的状态组件
            selected_topic_id = gr.State(value=None)
            
            # 设置事件处理
            self._setup_events(
                user_input, send_btn, chat_display, topics_display,
                new_topic_btn, clear_history_btn, export_history_btn, import_history_btn,
                clear_chat_btn, model_selector, switch_model_btn, refresh_topics_btn,
                selected_topic_id, current_topic_title,
                topic_selector, load_topic_btn
            )
        
        return interface
    
    def _setup_events(self, user_input, send_btn, chat_display, topics_display,
                     new_topic_btn, clear_history_btn, export_history_btn, import_history_btn,
                     clear_chat_btn, model_selector, switch_model_btn, refresh_topics_btn,
                     selected_topic_id, current_topic_title,
                     topic_selector, load_topic_btn):
        """设置事件处理"""
        
        # 发送消息 - 使用流式输出
        send_btn.click(
            fn=self._send_message_stream,
            inputs=[user_input, selected_topic_id],
            outputs=[user_input, chat_display, topics_display, topic_selector, selected_topic_id, current_topic_title]
        )
        
        # 回车发送 - 使用流式输出
        user_input.submit(
            fn=self._send_message_stream,
            inputs=[user_input, selected_topic_id],
            outputs=[user_input, chat_display, topics_display, topic_selector, selected_topic_id, current_topic_title]
        )
        
        # 新建话题
        new_topic_btn.click(
            fn=self._new_topic,
            outputs=[chat_display, topics_display, topic_selector, selected_topic_id, current_topic_title]
        )
        
        # 清空对话
        clear_chat_btn.click(
            fn=self._clear_chat,
            outputs=[chat_display, selected_topic_id, current_topic_title]
        )
        
        # 清空历史
        clear_history_btn.click(
            fn=self._clear_history,
            outputs=[selected_topic_id, current_topic_title]
        )
        
        # 导出历史
        export_history_btn.click(
            fn=self._export_history
        )
        
        # 导入历史
        import_history_btn.click(
            fn=self._import_history,
            outputs=[selected_topic_id]
        )
        
        # 刷新话题
        refresh_topics_btn.click(
            fn=self._refresh_topic_selector,
            outputs=[topic_selector, topics_display]
        )
        
        # 切换模型
        switch_model_btn.click(
            fn=self._switch_model,
            inputs=[model_selector]
        )
        
        # 话题选择功能
        load_topic_btn.click(
            fn=self._load_selected_topic,
            inputs=[topic_selector],
            outputs=[chat_display, current_topic_title, selected_topic_id]
        )
    
    def _handle_topic_click(self, topic_id: str) -> tuple:
        """处理话题点击事件"""
        if not topic_id:
            return gr.update(), gr.update(), None
        
        try:
            chat_html = self._render_topic_conversation(topic_id)
            topic_title = self._get_topic_title(topic_id)
            
            return (
                chat_html,
                f"**当前话题：** {topic_title}",
                topic_id
            )
        except Exception as e:
            logger.error(f"加载话题失败: {str(e)}")
            return (
                "<div class='chat-container'><p style='color: red;'>加载话题失败</p></div>",
                "**当前话题：** 错误",
                None
            )
    
    def _refresh_topics(self) -> str:
        """刷新话题列表"""
        logger.info("话题列表已刷新")
        return self._render_topics()
    
    def _send_message_stream(self, user_input: str, current_topic_id: Optional[str]) -> Generator:
        """流式发送消息处理"""
        if not user_input.strip():
            yield "", gr.update(), gr.update(), gr.update(), current_topic_id, gr.update()
            return

        if not current_topic_id:
            yield (
                user_input,
                "<div class='chat-container'><p style='color: red;'>请先点击 '➕ 新话题' 来开始一个新的对话。</p></div>",
                gr.update(), gr.update(), None, gr.update()
            )
            return
            
        try:
            topic_id = current_topic_id
            topic = self._get_topic_by_id(topic_id)
            is_first_message = topic and topic["title"] == "新建话题" and not topic["messages"]

            current_history = self._get_topic_history(topic_id)
            stream_generator = self.qa_agent.run_stream(user_input, current_history)
            
            for stream_result in stream_generator:
                status = stream_result.get("status", "processing")
                response_text = stream_result.get("response", "")
                
                if status == "processing":
                    chat_html = self._render_topic_conversation_with_streaming(
                        topic_id, user_input, response_text
                    )
                    yield ("", chat_html, gr.update(), gr.update(), topic_id, gr.update())
                
                elif status == "completed":
                    self._save_to_topic(topic_id, user_input, stream_result)
                    
                    final_chat_html = self._render_topic_conversation(topic_id)
                    final_topic_title_str = topic['title']
                    
                    if is_first_message:
                        logger.info(f"为话题 {topic_id} 生成新标题...")
                        new_title = self._generate_topic_title(user_input)
                        self._update_topic_title(topic_id, new_title)
                        final_topic_title_str = new_title
                        logger.info(f"话题 {topic_id} 标题更新为: {new_title}")

                    updated_topics_html = self._render_topics()
                    updated_selector = gr.Dropdown(choices=self._get_topic_choices(), value=topic_id)
                    final_topic_title = f"**当前话题：** {final_topic_title_str}"

                    yield ("", final_chat_html, updated_topics_html, updated_selector, topic_id, final_topic_title)
                
                elif status == "error":
                    error_chat_html = self._render_topic_conversation_with_streaming(
                        topic_id, user_input, f"发生错误: {response_text}"
                    )
                    yield ("", error_chat_html, gr.update(), gr.update(), topic_id, gr.update())
                    break
            
        except Exception as e:
            logger.error(f"流式发送消息失败: {str(e)}")
            error_msg = f"抱歉，处理您的消息时出现错误: {str(e)}"
            yield ("", f"<div class='chat-container'><p style='color: red;'>{error_msg}</p></div>", gr.update(), gr.update(), current_topic_id, gr.update())
    
    def _get_topic_by_id(self, topic_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取话题"""
        for topic in self.topics:
            if topic["id"] == topic_id:
                return topic
        return None

    def _update_topic_title(self, topic_id: str, new_title: str) -> None:
        """更新话题标题"""
        topic = self._get_topic_by_id(topic_id)
        if topic:
            topic["title"] = new_title
            self._save_history()
    
    def _render_topic_conversation_with_streaming(self, topic_id: str, current_user_input: str, streaming_response: str) -> str:
        """渲染包含流式响应的话题对话内容"""
        topic = None
        for t in self.topics:
            if t["id"] == topic_id:
                topic = t
                break
        
        html = "<div class='chat-container'>"
        
        # 渲染历史消息
        if topic and topic["messages"]:
            for msg in topic["messages"]:
                # 用户消息
                user_html = f"""
                <div class='message user-message'>
                    <strong>您:</strong><br>
                    {msg['user_input'].replace('<', '&lt;').replace('>', '&gt;')}
                </div>
                """
                
                # 助手回复 - 转换为HTML格式的Markdown
                assistant_response_html = markdown.markdown(
                    msg['assistant_response'],
                    extensions=['tables', 'fenced_code', 'codehilite']
                )
                assistant_html = f"""
                <div class='message assistant-message'>
                    <strong>医疗助手:</strong><br>
                    {assistant_response_html}
                </div>
                """
                
                html += user_html + assistant_html
        
        # 添加当前用户输入
        current_user_html = f"""
        <div class='message user-message'>
            <strong>您:</strong><br>
            {current_user_input.replace('<', '&lt;').replace('>', '&gt;')}
        </div>
        """
        
        # 添加流式响应
        if streaming_response == "正在思考中...":
            streaming_html = f"""
            <div class='message assistant-message'>
                <strong>医疗助手:</strong><br>
                {streaming_response} <span class='streaming-indicator'></span>
            </div>
            """
        else:
            # 转换流式响应为HTML格式的Markdown
            streaming_response_html = markdown.markdown(
                streaming_response,
                extensions=['tables', 'fenced_code', 'codehilite']
            )
            streaming_html = f"""
            <div class='message assistant-message'>
                <strong>医疗助手:</strong><br>
                {streaming_response_html}
            </div>
            """
        
        html += current_user_html + streaming_html + "</div>"
        return html
    
    def _create_new_topic(self, first_question: str) -> str:
        """创建新话题"""
        topic_id = f"topic_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        topic_title = self._generate_topic_title(first_question)
        
        new_topic = {
            "id": topic_id,
            "title": topic_title,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "messages": []
        }
        
        self.topics.append(new_topic)
        self._save_history()
        
        return topic_id
    
    def _save_to_topic(self, topic_id: str, user_input: str, result: Dict[str, Any]) -> None:
        """保存对话到指定话题"""
        # 找到对应话题
        topic = None
        for t in self.topics:
            if t["id"] == topic_id:
                topic = t
                break
        
        if not topic:
            return
        
        # 添加消息
        message = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "assistant_response": result["response"],
            "agent_type": result.get("agent_type", ""),
            "model_type": result.get("model_type", ""),
            "tools_used": result.get("tools_used", []),
            "metadata": result.get("metadata", {})
        }
        
        topic["messages"].append(message)
        topic["updated_at"] = datetime.now().isoformat()
        
        self._save_history()
    
    def _get_topic_history(self, topic_id: str) -> List[Dict[str, str]]:
        """获取话题的聊天历史"""
        topic = None
        for t in self.topics:
            if t["id"] == topic_id:
                topic = t
                break
        
        if not topic:
            return []
        
        history = []
        for msg in topic["messages"]:
            history.append({"role": "user", "content": msg["user_input"]})
            history.append({"role": "assistant", "content": msg["assistant_response"]})
        
        return history
    
    def _get_topic_title(self, topic_id: str) -> str:
        """获取话题标题"""
        for topic in self.topics:
            if topic["id"] == topic_id:
                return topic["title"]
        return "未知话题"
    
    def _render_topics(self) -> str:
        """渲染话题列表"""
        if not self.topics:
            return "<div class='history-container'><p style='text-align: center; color: #666;'>暂无历史话题</p></div>"
        
        html = "<div class='history-container'>"
        
        # 按更新时间倒序排列
        sorted_topics = sorted(self.topics, key=lambda x: x["updated_at"], reverse=True)
        
        for topic in sorted_topics:
            updated_time = datetime.fromisoformat(topic["updated_at"]).strftime("%m-%d %H:%M")
            message_count = len(topic["messages"])
            
            html += f"""
            <div class='topic-item' data-topic-id='{topic["id"]}'>
                <div class='topic-title'>{topic["title"]}</div>
                <div class='topic-info'>{updated_time} · {message_count}条对话</div>
            </div>
            """
        
        html += "</div>"
        return html
    
    def _render_topic_conversation(self, topic_id: str) -> str:
        """渲染话题的对话内容"""
        for topic in self.topics:
            if topic["id"] == topic_id:
                if not topic["messages"]:
                    return "<div class='chat-container'><p style='text-align: center; color: #666;'>该话题暂无对话记录</p></div>"
                
                chat_html = "<div class='chat-container'>"
                for message in topic["messages"]:
                    # 用户消息
                    chat_html += f"""
                    <div class="message user-message">
                        <strong>👤 您：</strong><br>{message["user_input"]}
                    </div>
                    """
                    # 助手回复 - 处理Markdown格式
                    content_html = self._markdown_to_html(message["assistant_response"])
                    chat_html += f"""
                    <div class="message assistant-message">
                        <strong>🤖 医疗助手：</strong><br>{content_html}
                    </div>
                    """
                chat_html += "</div>"
                return chat_html
        
        return "<div class='chat-container'><p style='color: red;'>找不到该话题</p></div>"
    
    def _new_topic(self) -> tuple:
        """新建话题, 在历史记录中创建一个占位符"""
        logger.info("创建新话题...")
        topic_id = str(uuid.uuid4())
        
        new_topic = {
            "id": topic_id,
            "title": "新建话题",  # 占位符标题
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "messages": []
        }
        
        self.topics.insert(0, new_topic)
        self._save_history()
        
        # 更新UI
        return (
            "<div class='chat-container'><p style='text-align: center; color: #666;'>新话题已创建，请开始您的对话...</p></div>",
            self._render_topics(),
            gr.Dropdown(choices=self._get_topic_choices(), value=topic_id),
            topic_id,
            "**当前话题：** 新建话题"
        )
    
    def _clear_chat(self) -> tuple:
        """清空当前对话"""
        return (
            "<div class='chat-container'><p style='text-align: center; color: #666;'>开始您的医疗咨询...</p></div>",
            None,
            "**当前话题：** 新对话"
        )
    
    def _clear_history(self) -> tuple:
        """清空所有历史"""
        self.topics = []
        self._save_history()
        return (
            None,
            "**当前话题：** 新对话"
        )
    
    def _load_topic(self, topic_id: str) -> tuple:
        """加载指定话题"""
        if not topic_id:
            return gr.update(), gr.update()
        
        chat_html = self._render_topic_conversation(topic_id)
        topic_title = self._get_topic_title(topic_id)
        
        return chat_html, f"**当前话题：** {topic_title}"
    
    def _export_history(self) -> None:
        """导出历史记录"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.topics, f, ensure_ascii=False, indent=2)
            logger.info(f"历史记录已导出到: {self.history_file}")
        except Exception as e:
            logger.error(f"导出历史记录失败: {str(e)}")
    
    def _import_history(self) -> str:
        """导入历史记录"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    self.topics = json.load(f)
                logger.info(f"历史记录已从 {self.history_file} 导入")
            return None
        except Exception as e:
            logger.error(f"导入历史记录失败: {str(e)}")
            return None
    
    def _switch_model(self, model_type: str) -> None:
        """切换模型"""
        try:
            self.qa_agent.switch_model(model_type)
            self.model_type = model_type
            logger.info(f"模型已切换到: {model_type}")
        except Exception as e:
            logger.error(f"切换模型失败: {str(e)}")
    
    def _load_history(self) -> None:
        """加载历史记录"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 检查是否是新的话题格式
                    if isinstance(data, list) and data and all("id" in item and "title" in item and "messages" in item for item in data):
                        self.topics = data
                        logger.info(f"历史记录已加载: {len(self.topics)} 个话题")
                    else:
                        # 如果不是话题格式，重置为空
                        self.topics = []
                        logger.info("历史记录格式不匹配，已重置")
            else:
                self.topics = []
                logger.info("未找到历史记录文件，已初始化为空")
        except Exception as e:
            logger.error(f"加载历史记录失败: {str(e)}")
            self.topics = []
    
    def _save_history(self) -> None:
        """保存历史记录"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.topics, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存历史记录失败: {str(e)}")
    
    def _markdown_to_html(self, text: str) -> str:
        """将Markdown文本转换为HTML"""
        try:
            return markdown.markdown(
                text,
                extensions=['tables', 'fenced_code', 'codehilite', 'nl2br']
            )
        except Exception as e:
            logger.error(f"Markdown转换失败: {str(e)}")
            # 如果转换失败，返回原始文本并转义HTML字符
            return text.replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br>')
    
    def _clear_old_history(self) -> None:
        """清除旧格式的历史记录"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 检查是否是旧格式（数组中的元素有id, timestamp等字段但没有话题结构）
                    if isinstance(data, list) and data and not any("id" in item and "title" in item and "messages" in item for item in data):
                        logger.info("检测到旧格式历史记录，正在清除...")
                        os.remove(self.history_file)
                        logger.info("旧格式历史记录已清除")
        except Exception as e:
            logger.error(f"清除旧历史记录失败: {str(e)}")
    
    def _refresh_topic_selector(self) -> tuple:
        """刷新话题选择器"""
        return (
            gr.Dropdown(choices=self._get_topic_choices(), value=None),
            self._render_topics()
        )
    
    def launch(self, **kwargs) -> None:
        """启动界面"""
        self.interface.launch(**kwargs)
    
    def _get_topic_choices(self) -> list:
        """获取话题选择器的选项"""
        if not self.topics:
            return []
        
        choices = []
        sorted_topics = sorted(self.topics, key=lambda x: x["updated_at"], reverse=True)
        
        for topic in sorted_topics:
            updated_time = datetime.fromisoformat(topic["updated_at"]).strftime("%m-%d %H:%M")
            message_count = len(topic["messages"])
            choice_text = f"{topic['title']} ({updated_time}, {message_count}条对话)"
            choices.append((choice_text, topic["id"]))
        
        return choices
    
    def _load_selected_topic(self, topic_id: str) -> tuple:
        """加载选中的话题"""
        if not topic_id:
            return (
                "<div class='chat-container'><p style='text-align: center; color: #666;'>请选择一个话题</p></div>",
                "**当前话题：** 未选择",
                None
            )
        
        return self._handle_topic_click(topic_id)


def main():
    """主函数"""
    # 创建界面实例
    interface = GradioInterface(
        model_type="api",
        verbose=True
    )
    
    # 启动界面
    interface.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        debug=True
    )


if __name__ == "__main__":
    main() 