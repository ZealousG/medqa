import os
from dotenv import load_dotenv

load_dotenv(".env")

class ModelConfig:
    def __init__(self):
        # 选择模型类型  
        self.use_api = True  # 是否使用API
        self.use_fastllm = False  # 是否使用FastLLM加速  
        self.use_vllm = False  # 是否使用VLLM加速 

        # 本地模型配置（可在.env中设置 MODEL_PATH 和 DEVICE）
        self.model_path = os.getenv("MODEL_PATH", "/mnt/d/data/LLMs/Qwen/Qwen2.5-1.5B")  # 本地模型路径  
        self.device = "cuda:0"  # 设备  

        # volcengine API配置  
        self.volc_api_key = os.getenv("VOLC_API_KEY")
        self.volc_api_base = os.getenv("VOLC_API_BASE")
        self.volc_model_name = os.getenv("VOLC_MODEL")

        # 知识库配置  
        self.retriever_type = "knn"  # 检索类型：knn, similarity, bm25, l2  
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL")  
        self.index_path = "data/indices/medical_books"  
        self.top_k = 5  # 默认检索数量  

        # 推理配置  
        self.max_length = 2048  # 最大输入长度（兼容旧参数名）
        self.temperature = 0.7  # 采样温度
        self.top_p = 0.9  # nucleus sampling

        # 服务配置  
        self.host = "0.0.0.0"  
        self.port = 8000  
        self.debug = False  
        self.verbose = False  