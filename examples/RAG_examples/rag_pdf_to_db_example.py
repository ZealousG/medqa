"""
RAG批量PDF解析与入库示例
将/data/doc下所有PDF解析、分块并合并为一个知识库，建立RAG检索索引。
"""
import os
from src.RAG.preprocess.document_loader import DocumentLoader
from src.RAG.preprocess.document_processor import DocumentProcessor
from src.RAG.rag_pipeline import RAGPipeline
from src.RAG.database_manager.database_manager import KnowledgeBaseManager
from langchain.schema import Document
from src.configs.configs import ModelConfig

PDF_DIR = "data/doc"
KB_NAME = "medical_books"


def main():
    # 0. 加载配置
    config = ModelConfig()
    index_dir = os.path.dirname(config.index_path) if config.index_path else "knowledge_base/indices"
    embedding_model_name = config.embedding_model_name
    retriever_type = config.retriever_type

    # 1. 批量加载PDF
    loader = DocumentLoader(chunk_size=500, chunk_overlap=50)
    print(f"正在加载{PDF_DIR}下所有PDF...")
    documents = loader.load_from_directory(PDF_DIR, recursive=False)
    print(f"共加载文档: {len(documents)}")

    # 2. 文本清洗与分块
    processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
    print("正在清洗与分块...")
    chunks = processor.process_documents(documents)
    print(f"分块后总块数: {len(chunks)}")

    # 3. 创建知识库（如不存在）
    kb_manager = KnowledgeBaseManager(index_dir=index_dir)
    if KB_NAME not in [kb["name"] for kb in kb_manager.list_knowledge_bases()]:
        kb_manager.create_knowledge_base(KB_NAME, description="医学PDF合集")
    index_save_dir = os.path.join(index_dir, KB_NAME)

    # 4. 建立RAG检索索引
    print("正在建立RAG检索索引...")
    rag_pipeline = RAGPipeline(
        retriever_type=retriever_type,
        embedding_model_name=embedding_model_name
    )
    # 构建Document对象列表
    doc_objs = [Document(page_content=chunk.page_content, metadata=chunk.metadata) for chunk in chunks]
    # 添加到KNNRetriever
    rag_pipeline.retriever.add_documents(doc_objs)
    # 保存索引
    rag_pipeline.retriever.save(index_save_dir)
    print(f"索引已保存到: {index_save_dir}")

    # 5. 更新知识库文档数
    kb_manager.update_document_count(KB_NAME, len(chunks))
    print(f"知识库 {KB_NAME} 文档块数: {len(chunks)}")

    print("处理完成！")

if __name__ == "__main__":
    main() 