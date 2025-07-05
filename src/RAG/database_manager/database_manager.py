"""
知识库管理器
负责管理知识库的创建、删除和索引操作
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from src.RAG.preprocess.embedding_manager import EmbeddingManager

logger = logging.getLogger(__name__)

class KnowledgeBaseManager:
    """知识库管理器"""
    
    def __init__(self, index_dir: str = "knowledge_base/indices"):
        """
        初始化知识库管理器
        
        Args:
            index_dir: 索引目录路径
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # 知识库配置文件路径
        self.config_file = self.index_dir / "kb_config.json"
        self.knowledge_bases = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载知识库配置"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"加载知识库配置失败: {e}")
                return {}
        return {}
    
    def _save_config(self) -> None:
        """保存知识库配置"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge_bases, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存知识库配置失败: {e}")
    
    def initialize(self) -> None:
        """初始化知识库管理器"""
        logger.info("初始化知识库管理器")
        # 确保索引目录存在
        self.index_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"知识库索引目录: {self.index_dir}")
    
    def create_knowledge_base(self, kb_name: str, description: str = "") -> bool:
        """
        创建新知识库
        
        Args:
            kb_name: 知识库名称
            description: 知识库描述
            
        Returns:
            是否创建成功
        """
        try:
            if kb_name in self.knowledge_bases:
                logger.warning(f"知识库 {kb_name} 已存在")
                return False
            
            # 创建知识库目录
            kb_dir = self.index_dir / kb_name
            kb_dir.mkdir(exist_ok=True)
            
            # 保存知识库信息
            self.knowledge_bases[kb_name] = {
                "name": kb_name,
                "description": description,
                "created_at": str(Path().stat().st_ctime),
                "index_path": str(kb_dir),
                "document_count": 0
            }
            
            self._save_config()
            logger.info(f"创建知识库成功: {kb_name}")
            return True
            
        except Exception as e:
            logger.error(f"创建知识库失败: {e}")
            return False
    
    def delete_knowledge_base(self, kb_name: str) -> bool:
        """
        删除知识库
        
        Args:
            kb_name: 知识库名称
            
        Returns:
            是否删除成功
        """
        try:
            if kb_name not in self.knowledge_bases:
                logger.warning(f"知识库 {kb_name} 不存在")
                return False
            
            # 删除知识库目录
            kb_dir = self.index_dir / kb_name
            if kb_dir.exists():
                import shutil
                shutil.rmtree(kb_dir)
            
            # 从配置中移除
            del self.knowledge_bases[kb_name]
            self._save_config()
            
            logger.info(f"删除知识库成功: {kb_name}")
            return True
            
        except Exception as e:
            logger.error(f"删除知识库失败: {e}")
            return False
    
    def get_index_path(self, kb_name: str) -> Optional[Path]:
        """
        获取知识库索引路径
        
        Args:
            kb_name: 知识库名称
            
        Returns:
            索引路径，如果不存在则返回None
        """
        if kb_name not in self.knowledge_bases:
            return None
        
        kb_info = self.knowledge_bases[kb_name]
        index_path = Path(kb_info.get("index_path", ""))
        
        if not index_path.exists():
            return None
        
        return index_path
    
    def list_knowledge_bases(self) -> List[Dict[str, Any]]:
        """
        获取知识库列表
        
        Returns:
            知识库信息列表
        """
        return list(self.knowledge_bases.values())
    
    def get_knowledge_base_info(self, kb_name: str) -> Optional[Dict[str, Any]]:
        """
        获取知识库信息
        
        Args:
            kb_name: 知识库名称
            
        Returns:
            知识库信息，如果不存在则返回None
        """
        return self.knowledge_bases.get(kb_name)
    
    def update_document_count(self, kb_name: str, count: int) -> bool:
        """
        更新知识库文档数量
        
        Args:
            kb_name: 知识库名称
            count: 文档数量
            
        Returns:
            是否更新成功
        """
        try:
            if kb_name not in self.knowledge_bases:
                return False
            
            self.knowledge_bases[kb_name]["document_count"] = count
            self._save_config()
            return True
            
        except Exception as e:
            logger.error(f"更新文档数量失败: {e}")
            return False 