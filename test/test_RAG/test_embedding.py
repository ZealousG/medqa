"""
测试embedding模型是否正常工作
"""
from src.configs.configs import ModelConfig
from src.RAG.preprocess.embedding_manager import EmbeddingManager

def main():
    config = ModelConfig()
    print("embedding_model_name:", config.embedding_model_name)
    
    # 测试embedding
    try:
        embedding_manager = EmbeddingManager(
            embedding_model_name=config.embedding_model_name
        )
        
        # 测试embedding一个简单句子
        test_text = "什么是高血压？"
        embedding = embedding_manager.embed_query(test_text)
        
        print(f"embedding成功！维度: {len(embedding)}")
        print(f"前5个值: {embedding[:5]}")
        
    except Exception as e:
        print(f"embedding失败: {e}")

if __name__ == "__main__":
    main() 