# src/rag/context_builder.py

import os
import re
from typing import Dict, List, Optional, Union, Any, Tuple
import torch
from sentence_transformers import CrossEncoder
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class ContextBuilder:
    """
    上下文构建器，负责从检索结果构建增强上下文
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        reranker_model_name: Optional[str] = None,
        max_context_length: int = 4000,
        format_template: Optional[str] = None
    ):
        """
        初始化上下文构建器
        
        Args:
            chunk_size: 上下文块大小
            chunk_overlap: 上下文块重叠大小
            reranker_model_name: 重排序模型名称
            max_context_length: 最大上下文长度
            format_template: 格式化模板
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_context_length = max_context_length
        
        # 设置格式化模板
        if format_template:
            self.format_template = format_template
        else:
            self.format_template = (
                "请根据以下信息回答问题。如果无法从提供的信息中找到答案，请基于可靠的医学知识回答，"
                "并注明这是基于一般医学知识的回答。\n\n相关信息：\n{context}\n\n问题：{query}\n\n回答："
            )
        
        # 初始化重排序模型
        self.reranker = None
        if reranker_model_name:
            try:
                self.reranker = CrossEncoder(reranker_model_name)
                logger.info(f"已加载重排序模型: {reranker_model_name}")
            except Exception as e:
                logger.error(f"加载重排序模型失败: {e}")
    
    def split_document(self, text: str) -> List[str]:
        """
        将文档分割成小块
        
        Args:
            text: 文档文本
            
        Returns:
            文本块列表
        """
        # 先按段落分割
        paragraphs = re.split(r'\n+', text)
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # 如果段落本身就超过chunk_size，需要进一步分割
            if len(paragraph) > self.chunk_size:
                # 如果当前chunk不为空，先添加到chunks
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                
                # 按句子分割长段落
                sentences = re.split(r'([.。!！?？;；])', paragraph)
                current_sentence = ""
                
                for i in range(0, len(sentences), 2):
                    if i+1 < len(sentences):
                        sentence = sentences[i] + sentences[i+1]
                    else:
                        sentence = sentences[i]
                    
                    if len(current_sentence) + len(sentence) <= self.chunk_size:
                        current_sentence += sentence
                    else:
                        if current_sentence:
                            chunks.append(current_sentence)
                        current_sentence = sentence
                
                if current_sentence:
                    chunks.append(current_sentence)
            
            # 正常处理不超过chunk_size的段落
            elif len(current_chunk) + len(paragraph) <= self.chunk_size:
                current_chunk += "\n" + paragraph if current_chunk else paragraph
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = paragraph
        
        # 添加最后一个chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def rerank_documents(
        self,
        query: str,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        使用重排序模型对文档进行重新排序
        
        Args:
            query: 查询文本
            documents: 文档列表
            
        Returns:
            重新排序后的文档列表
        """
        if not self.reranker:
            logger.warning("未初始化重排序模型，跳过重排序")
            return documents
        
        document_texts = [doc["text"] for doc in documents]
        
        # 准备输入对
        pairs = [(query, text) for text in document_texts]
        
        # 计算相关性分数
        scores = self.reranker.predict(pairs)
        
        # 更新文档分数和排序
        scored_documents = []
        for i, (doc, score) in enumerate(zip(documents, scores)):
            doc_copy = doc.copy()
            doc_copy["rerank_score"] = float(score)
            scored_documents.append(doc_copy)
        
        # 根据分数排序
        sorted_documents = sorted(scored_documents, key=lambda x: x["rerank_score"], reverse=True)
        
        return sorted_documents
    
    def build_context(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        use_reranker: bool = False
    ) -> str:
        """
        从文档列表构建上下文
        
        Args:
            query: 查询文本
            documents: 文档列表
            use_reranker: 是否使用重排序
            
        Returns:
            构建的上下文
        """
        if not documents:
            logger.warning("没有文档用于构建上下文")
            return ""
        
        # 如果要使用重排序
        if use_reranker and self.reranker:
            documents = self.rerank_documents(query, documents)
        
        # 构建上下文
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(documents):
            # 提取文档文本
            text = doc["text"]
            
            # 如果文档过长，分割成小块
            if len(text) > self.chunk_size:
                chunks = self.split_document(text)
                
                # 添加每个块作为单独的部分
                for j, chunk in enumerate(chunks):
                    if current_length + len(chunk) + 100 <= self.max_context_length:  # 100是额外标记的空间
                        score_info = ""
                        if "score" in doc:
                            score_info = f" (相关度: {doc['score']:.2f})"
                        elif "rerank_score" in doc:
                            score_info = f" (相关度: {doc['rerank_score']:.2f})"
                        
                        source_info = ""
                        if "metadata" in doc and doc["metadata"]:
                            if "source" in doc["metadata"]:
                                source_info = f" [来源: {doc['metadata']['source']}]"
                        
                        # 构建标记
                        part_header = f"[文档 {i+1}-{j+1}]{score_info}{source_info}:\n"
                        part = f"{part_header}{chunk}\n\n"
                        
                        context_parts.append(part)
                        current_length += len(part)
                    else:
                        # 上下文已满
                        break
            else:
                # 直接添加整个文档
                if current_length + len(text) + 100 <= self.max_context_length:
                    score_info = ""
                    if "score" in doc:
                        score_info = f" (相关度: {doc['score']:.2f})"
                    elif "rerank_score" in doc:
                        score_info = f" (相关度: {doc['rerank_score']:.2f})"
                    
                    source_info = ""
                    if "metadata" in doc and doc["metadata"]:
                        if "source" in doc["metadata"]:
                            source_info = f" [来源: {doc['metadata']['source']}]"
                    
                    # 构建标记
                    part_header = f"[文档 {i+1}]{score_info}{source_info}:\n"
                    part = f"{part_header}{text}\n\n"
                    
                    context_parts.append(part)
                    current_length += len(part)
                else:
                    # 上下文已满
                    break
        
        # 拼接上下文
        context = "".join(context_parts)
        
        # 截断过长的上下文
        if len(context) > self.max_context_length:
            context = context[:self.max_context_length]
            logger.warning(f"上下文过长，已截断至 {self.max_context_length} 字符")
        
        return context.strip()
    
    def format_prompt(self, query: str, context: str) -> str:
        """
        格式化提示
        
        Args:
            query: 查询文本
            context: 上下文
            
        Returns:
            格式化后的提示
        """
        return self.format_template.format(query=query, context=context)
    
    def extract_key_information(self, context: str, query: str) -> str:
        """
        从上下文中提取与查询最相关的关键信息
        
        Args:
            context: 上下文
            query: 查询
            
        Returns:
            提取的关键信息
        """
        if not context:
            return ""
        
        # 分割上下文为不同的文档块
        doc_pattern = r'\[文档 \d+(?:-\d+)?\](?:\s*\(相关度:[0-9.]+\))?(?:\s*\[来源: [^\]]+\])?:\n(.*?)(?=\n\n\[文档 \d+|\Z)'
        doc_matches = re.finditer(doc_pattern, context, re.DOTALL)
        
        # 提取文档内容
        doc_contents = []
        for match in doc_matches:
            content = match.group(1).strip()
            if content:
                doc_contents.append(content)
        
        # 如果无法分割，直接返回上下文
        if not doc_contents:
            return context
        
        # 如果有重排序模型，使用它来选择最相关的段落
        if self.reranker:
            # 准备输入对
            pairs = [(query, content) for content in doc_contents]
            
            # 计算相关性分数
            scores = self.reranker.predict(pairs)
            
            # 对段落按相关性排序
            sorted_contents = [content for _, content in sorted(
                zip(scores, doc_contents), 
                key=lambda x: x[0], 
                reverse=True
            )]
            
            # 选择最相关的内容，控制总长度
            key_info = ""
            current_length = 0
            
            for content in sorted_contents:
                if current_length + len(content) + 10 <= self.max_context_length // 2:
                    key_info += content + "\n\n"
                    current_length += len(content) + 2
                else:
                    break
            
            return key_info.strip()
        else:
            # 如果没有重排序模型，简单连接前几个文档内容
            combined_content = "\n\n".join(doc_contents[:3])
            
            # 如果内容过长，截断
            if len(combined_content) > self.max_context_length // 2:
                return combined_content[:self.max_context_length // 2] + "..."
            
            return combined_content