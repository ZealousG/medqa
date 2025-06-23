import gradio as gr
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.agents.QA_agent import QA_Agent
from src.configs.model_config import ModelConfig
from src.utils.logger import setup_logger

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
        
        # 历史记录存储
        self.chat_history: List[Dict[str, Any]] = []
        self.history_file = "chat_history.json"
        self._load_history()
        
        # 创建界面
        self.interface = self._create_interface()
    
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
                height: 600px;
                overflow-y: auto;
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 10px;
                background-color: #f5f5f5;
            }
            .message {
                margin: 10px 0;
                padding: 10px;
                border-radius: 8px;
                max-width: 80%;
            }
            .user-message {
                background-color: #007bff;
                color: white;
                margin-left: auto;
                text-align: right;
            }
            .assistant-message {
                background-color: #e9ecef;
                color: black;
                margin-right: auto;
            }
            .history-item {
                padding: 8px;
                margin: 5px 0;
                border-radius: 5px;
                cursor: pointer;
                border: 1px solid #ddd;
                background-color: white;
            }
            .history-item:hover {
                background-color: #f0f0f0;
            }
            .history-item.active {
                background-color: #007bff;
                color: white;
            }
            """
        ) as interface:
            
            gr.Markdown("# 🏥 医疗问答系统")
            gr.Markdown("### 专业的医疗咨询助手，为您提供诊断、治疗和预防建议")
            
            with gr.Row():
                # 左侧历史记录面板
                with gr.Column(scale=1):
                    gr.Markdown("### 📚 历史记录")
                    
                    # 历史记录列表
                    history_list = gr.List(
                        value=[],
                        label="历史对话",
                        elem_classes=["history-container"],
                        interactive=True
                    )
                    
                    # 历史记录操作按钮
                    with gr.Row():
                        clear_history_btn = gr.Button("🗑️ 清空历史", variant="secondary", size="sm")
                        export_history_btn = gr.Button("📤 导出历史", variant="secondary", size="sm")
                        import_history_btn = gr.Button("📥 导入历史", variant="secondary", size="sm")
                    
                    # 模型信息显示
                    model_info = gr.JSON(
                        value=self.qa_agent.get_model_info(),
                        label="模型信息"
                    )
                
                # 右侧对话面板
                with gr.Column(scale=2):
                    gr.Markdown("### 💬 对话区域")
                    
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
            
            # 设置事件处理
            self._setup_events(
                user_input, send_btn, chat_display, history_list,
                clear_history_btn, export_history_btn, import_history_btn,
                clear_chat_btn, model_selector, switch_model_btn, model_info
            )
        
        return interface
    
    def _setup_events(self, user_input, send_btn, chat_display, history_list,
                     clear_history_btn, export_history_btn, import_history_btn,
                     clear_chat_btn, model_selector, switch_model_btn, model_info):
        """设置事件处理"""
        
        # 发送消息
        send_btn.click(
            fn=self._send_message,
            inputs=[user_input, chat_display],
            outputs=[user_input, chat_display, history_list]
        )
        
        # 回车发送
        user_input.submit(
            fn=self._send_message,
            inputs=[user_input, chat_display],
            outputs=[user_input, chat_display, history_list]
        )
        
        # 清空对话
        clear_chat_btn.click(
            fn=self._clear_chat,
            outputs=[chat_display]
        )
        
        # 清空历史
        clear_history_btn.click(
            fn=self._clear_history,
            outputs=[history_list, chat_display]
        )
        
        # 导出历史
        export_history_btn.click(
            fn=self._export_history
        )
        
        # 导入历史
        import_history_btn.click(
            fn=self._import_history,
            outputs=[history_list]
        )
        
        # 切换模型
        switch_model_btn.click(
            fn=self._switch_model,
            inputs=[model_selector],
            outputs=[model_info]
        )
        
        # 选择历史记录
        history_list.select(
            fn=self._load_history_conversation,
            inputs=[history_list],
            outputs=[chat_display]
        )
    
    def _send_message(self, user_input: str, chat_display: str) -> tuple:
        """发送消息处理"""
        if not user_input.strip():
            return "", chat_display, gr.update()
        
        try:
            # 获取当前聊天历史
            current_history = self._parse_chat_display(chat_display)
            
            # 调用 QA Agent
            result = self.qa_agent.run(user_input, current_history)
            
            # 更新聊天显示
            new_chat_display = self._update_chat_display(
                chat_display, user_input, result["response"]
            )
            
            # 保存到历史记录
            self._save_to_history(user_input, result)
            
            # 更新历史记录列表
            history_list = self._get_history_list()
            
            return "", new_chat_display, gr.update(value=history_list)
            
        except Exception as e:
            logger.error(f"发送消息失败: {str(e)}")
            error_msg = f"抱歉，处理您的消息时出现错误: {str(e)}"
            new_chat_display = self._update_chat_display(chat_display, user_input, error_msg)
            return "", new_chat_display, gr.update()
    
    def _update_chat_display(self, current_display: str, user_msg: str, assistant_msg: str) -> str:
        """更新聊天显示"""
        # 解析当前显示内容
        if "开始您的医疗咨询" in current_display:
            chat_html = "<div class='chat-container'>"
        else:
            # 提取现有的聊天内容
            start_idx = current_display.find("<div class='chat-container'>")
            end_idx = current_display.find("</div>", start_idx)
            if start_idx != -1 and end_idx != -1:
                chat_html = current_display[start_idx:end_idx + 6]
            else:
                chat_html = "<div class='chat-container'>"
        
        # 添加新消息
        user_html = f"""
        <div class='message user-message'>
            <strong>您:</strong><br>
            {user_msg.replace('<', '&lt;').replace('>', '&gt;')}
        </div>
        """
        
        assistant_html = f"""
        <div class='message assistant-message'>
            <strong>医疗助手:</strong><br>
            {assistant_msg.replace('<', '&lt;').replace('>', '&gt;')}
        </div>
        """
        
        # 组合新的显示内容
        new_content = chat_html + user_html + assistant_html + "</div>"
        
        return new_content
    
    def _parse_chat_display(self, chat_display: str) -> List[Dict[str, str]]:
        """解析聊天显示内容为历史记录格式"""
        history = []
        
        # 简单的解析逻辑，实际项目中可能需要更复杂的HTML解析
        if "开始您的医疗咨询" in chat_display:
            return history
        
        # 这里简化处理，实际使用时可能需要更精确的HTML解析
        # 暂时返回空列表，让Agent处理当前对话
        return history
    
    def _save_to_history(self, user_input: str, result: Dict[str, Any]) -> None:
        """保存对话到历史记录"""
        history_item = {
            "id": len(self.chat_history) + 1,
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "assistant_response": result["response"],
            "agent_type": result.get("agent_type", ""),
            "model_type": result.get("model_type", ""),
            "tools_used": result.get("tools_used", []),
            "metadata": result.get("metadata", {})
        }
        
        self.chat_history.append(history_item)
        self._save_history()
    
    def _get_history_list(self) -> List[str]:
        """获取历史记录列表"""
        history_list = []
        for item in self.chat_history:
            # 截取用户输入的前50个字符作为标题
            title = item["user_input"][:50] + "..." if len(item["user_input"]) > 50 else item["user_input"]
            timestamp = datetime.fromisoformat(item["timestamp"]).strftime("%Y-%m-%d %H:%M")
            history_list.append(f"{timestamp} - {title}")
        
        return history_list
    
    def _load_history_conversation(self, evt: gr.SelectData) -> str:
        """加载历史对话"""
        if not evt.index or evt.index >= len(self.chat_history):
            return "<div class='chat-container'><p style='text-align: center; color: #666;'>开始您的医疗咨询...</p></div>"
        
        history_item = self.chat_history[evt.index]
        
        chat_html = "<div class='chat-container'>"
        user_html = f"""
        <div class='message user-message'>
            <strong>您:</strong><br>
            {history_item['user_input'].replace('<', '&lt;').replace('>', '&gt;')}
        </div>
        """
        
        assistant_html = f"""
        <div class='message assistant-message'>
            <strong>医疗助手:</strong><br>
            {history_item['assistant_response'].replace('<', '&lt;').replace('>', '&gt;')}
        </div>
        """
        
        return chat_html + user_html + assistant_html + "</div>"
    
    def _clear_chat(self) -> str:
        """清空当前对话"""
        return "<div class='chat-container'><p style='text-align: center; color: #666;'>开始您的医疗咨询...</p></div>"
    
    def _clear_history(self) -> tuple:
        """清空历史记录"""
        self.chat_history = []
        self._save_history()
        return gr.update(value=[]), self._clear_chat()
    
    def _export_history(self) -> None:
        """导出历史记录"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.chat_history, f, ensure_ascii=False, indent=2)
            logger.info(f"历史记录已导出到: {self.history_file}")
        except Exception as e:
            logger.error(f"导出历史记录失败: {str(e)}")
    
    def _import_history(self) -> List[str]:
        """导入历史记录"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    self.chat_history = json.load(f)
                logger.info(f"历史记录已从 {self.history_file} 导入")
            return self._get_history_list()
        except Exception as e:
            logger.error(f"导入历史记录失败: {str(e)}")
            return []
    
    def _switch_model(self, model_type: str) -> Dict[str, Any]:
        """切换模型"""
        try:
            self.qa_agent.switch_model(model_type)
            self.model_type = model_type
            return self.qa_agent.get_model_info()
        except Exception as e:
            logger.error(f"切换模型失败: {str(e)}")
            return self.qa_agent.get_model_info()
    
    def _load_history(self) -> None:
        """加载历史记录"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    self.chat_history = json.load(f)
                logger.info(f"历史记录已加载: {len(self.chat_history)} 条记录")
        except Exception as e:
            logger.error(f"加载历史记录失败: {str(e)}")
            self.chat_history = []
    
    def _save_history(self) -> None:
        """保存历史记录"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.chat_history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存历史记录失败: {str(e)}")
    
    def launch(self, **kwargs) -> None:
        """启动界面"""
        self.interface.launch(**kwargs)


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
        server_port=7860,
        share=False,
        debug=True
    )


if __name__ == "__main__":
    main() 