"""
调试KNNRetriever检索功能
"""
from src.configs.configs import ModelConfig
from src.RAG.retrieval.knn_retriever import KNNRetriever
from src.RAG.preprocess.embedding_manager import EmbeddingManager

def main():
    config = ModelConfig()
    
    # 1. 创建embedding管理器
    embedding_manager = EmbeddingManager(
        embedding_model_name=config.embedding_model_name
    )
    
    # 2. 创建KNN检索器
    retriever = KNNRetriever(embedding_manager=embedding_manager)
    
    # 3. 加载索引
    index_path = config.index_path
    print(f"加载索引: {index_path}")
    retriever.load(index_path)
    print(f"加载完成，文档数量: {len(retriever.documents)}")
    print(f"索引是否存在: {retriever.index is not None}")
    print(f"索引维度: {retriever.dimension}")
    print(f"默认score_threshold: {retriever.score_threshold}")
    
    # 4. 测试检索
    query = "什么是高血压？"
    print(f"\n测试查询: {query}")
    
    # 先测试embedding
    query_embedding = embedding_manager.embed_query(query)
    print(f"查询embedding维度: {len(query_embedding)}")
    
    # 执行检索 - 设置更低的阈值
    print("\n=== 使用score_threshold=0.0 ===")
    results = retriever.search(query, top_k=3, score_threshold=0.0)
    print(f"检索结果数量: {len(results)}")
    
    for i, (doc, score) in enumerate(results):
        print(f"结果{i+1}: 分数={score:.4f}")
        print(f"内容片段: {doc.page_content[:100]}...")
        print(f"元数据: {doc.metadata}")
        print()
    
    # 如果还是没结果，尝试负数阈值
    if len(results) == 0:
        print("\n=== 使用score_threshold=-1.0 ===")
        results = retriever.search(query, top_k=3, score_threshold=-1.0)
        print(f"检索结果数量: {len(results)}")
        
        for i, (doc, score) in enumerate(results):
            print(f"结果{i+1}: 分数={score:.4f}")
            print(f"内容片段: {doc.page_content[:100]}...")
            print()

if __name__ == "__main__":
    main() 