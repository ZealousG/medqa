"""
RAG流水线使用示例
展示如何正确设置模型和使用RAG流水线
"""

from src.RAG.rag_pipeline import RAGPipeline
from src.models.QA_model import create_qa_model
from src.configs.configs import ModelConfig

def main():
    """主函数：演示RAG流水线的使用"""
    
    # 1. 创建模型配置
    config = ModelConfig()
    
    # 2. 创建模型实例（使用现有的QA模型工厂）
    model = create_qa_model(model_type="api", model_config=config)
    
    # 3. 创建RAG流水线
    rag_pipeline = RAGPipeline(
        retriever_type="knn",
        embedding_model_name=config.embedding_model_name,
        top_k=config.top_k,
        model=model,  # 直接传入模型实例
        max_new_tokens=config.max_length,
        temperature=config.temperature,
        top_p=config.top_p
    )
    
    # 或者，如果需要在运行时动态设置模型
    # rag_pipeline = RAGPipeline(retriever_type="knn")
    # rag_pipeline.set_model(model)  # 使用set_model方法设置模型
    
    # 4. 使用RAG流水线进行查询
    query = "什么是高血压？"
    
    try:
        # 生成响应
        result = rag_pipeline.generate_response(query)
        
        print("查询:", query)
        print("响应:", result.get("response", "无响应"))
        print("生成时间:", result.get("metadata", {}).get("generation_time", 0))
        
    except Exception as e:
        print(f"生成响应时出错: {e}")

if __name__ == "__main__":
    main() 