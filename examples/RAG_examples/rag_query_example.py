"""
RAG知识库问答示例
演示如何加载已建好的知识库索引，结合大模型进行RAG问答。
"""
import os
import sys
print("工作目录:", os.getcwd())
print("Python 路径:", sys.path)

from src.RAG.rag_pipeline import RAGPipeline
from src.configs.configs import ModelConfig
from src.models.QA_model import create_qa_model

print("index_path:", "data/indices/medical_books")
print("目录内容:", os.listdir("data/indices/medical_books"))

def main():
    # 1. 加载配置
    config = ModelConfig()
    print("embedding_model_name:", config.embedding_model_name)
    print("volc_api_key:", config.volc_api_key)
    print("volc_api_base:", config.volc_api_base) 
    print("volc_model_name:", config.volc_model_name)
    
    # 2. 创建大模型实例
    model = create_qa_model(model_type="api" if config.use_api else "local", model_config=config)
    print("模型创建成功:", type(model))
    
    # 3. 创建RAG流水线，需要先加载索引
    rag_pipeline = RAGPipeline(
        retriever_type=config.retriever_type,
        embedding_model_name=config.embedding_model_name,
        model=model,
        top_k=config.top_k,
        max_new_tokens=config.max_length,
        temperature=config.temperature,
        top_p=config.top_p
    )
    
    # 4. 手动加载已保存的索引
    index_save_dir = config.index_path
    print("正在加载索引:", index_save_dir)
    rag_pipeline.retriever.load(index_save_dir)
    print("索引加载完成，文档数量:", len(rag_pipeline.retriever.documents))
    
    # 5. 查询
    query = "什么是高血压？"
    result = rag_pipeline.generate_response(query)
    print("查询：", query)
    print("RAG答案：", result.get("response", "无响应"))
    print("相关文档片段：")
    for doc in result.get("source_documents", []):
        if isinstance(doc, dict):
            print("-", doc.get("text", "")[:100] + "...")
        else:
            print("-", str(doc)[:100] + "...")

if __name__ == "__main__":
    main()