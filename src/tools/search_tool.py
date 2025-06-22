import requests
from typing import Dict, Any, Optional
import json
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from src.utils.logger import get_logger

logger = get_logger(__name__)

class SearchInput(BaseModel):
    query: str = Field(description="搜索查询，应该包含关键词和具体的医疗问题")
    num_results: Optional[int] = Field(default=5, description="返回结果数量")

class SearchTool(BaseTool):
    name: str = "搜索工具"
    description: str = "用于从互联网搜索医疗相关信息，可以提供最新的医学研究和治疗方法"
    args_schema: type[BaseModel] = SearchInput
    search_api_key: Optional[str] = None
    search_engine_id: Optional[str] = None

    def __init__(self, search_api_key: Optional[str] = None, search_engine_id: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.search_api_key = search_api_key
        self.search_engine_id = search_engine_id

    def _run(self, query: str, num_results: int = 5) -> str:
        if not self.search_api_key or not self.search_engine_id:
            return "搜索工具未配置API密钥和搜索引擎ID，无法执行搜索"
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": self.search_api_key,
                "cx": self.search_engine_id,
                "q": query,
                "num": min(num_results, 10)
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            search_results = response.json()
            if "items" not in search_results:
                return "没有找到相关结果"
            results_text = f"搜索查询 '{query}' 的结果:\n\n"
            for i, item in enumerate(search_results["items"][:num_results], 1):
                title = item.get("title", "无标题")
                link = item.get("link", "无链接")
                snippet = item.get("snippet", "无摘要")
                results_text += f"{i}. {title}\n"
                results_text += f"   链接: {link}\n"
                results_text += f"   摘要: {snippet}\n\n"
            return results_text
        except requests.RequestException as e:
            logger.error(f"搜索请求失败: {str(e)}")
            return f"搜索失败: {str(e)}"
        except Exception as e:
            logger.error(f"搜索过程中发生错误: {str(e)}")
            return f"搜索错误: {str(e)}"

    async def _arun(self, query: str, num_results: int = 5) -> str:
        return self._run(query, num_results)

class BingSearchTool(SearchTool):
    """使用Bing API的搜索工具"""
    
    def __init__(
        self, 
        subscription_key: Optional[str] = None,
        name: str = "Bing搜索",
        description: str = "使用Bing搜索医疗相关信息，获取权威医学网站的最新内容"
    ) -> None:
        """
        初始化Bing搜索工具
        
        Args:
            subscription_key: Bing搜索API订阅密钥
            name: 工具名称
            description: 工具描述
        """
        super().__init__(name=name, description=description)
        self.subscription_key = subscription_key
    
    def _run(self, query: str, num_results: int = 5) -> str:
        """
        执行Bing搜索
        
        Args:
            query: 搜索查询
            num_results: 返回结果数量
            
        Returns:
            搜索结果文本
        """
        if not self.subscription_key:
            return "Bing搜索工具未配置订阅密钥，无法执行搜索"
        
        try:
            # 使用Bing Search API
            url = "https://api.bing.microsoft.com/v7.0/search"
            headers = {"Ocp-Apim-Subscription-Key": self.subscription_key}
            params = {
                "q": query,
                "count": num_results,
                "responseFilter": "Webpages",
                "textFormat": "Raw"
            }
            
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            search_results = response.json()
            
            if "webPages" not in search_results or "value" not in search_results["webPages"]:
                return "没有找到相关结果"
            
            results_text = f"Bing搜索 '{query}' 的结果:\n\n"
            
            for i, item in enumerate(search_results["webPages"]["value"][:num_results], 1):
                title = item.get("name", "无标题")
                link = item.get("url", "无链接")
                snippet = item.get("snippet", "无摘要")
                
                results_text += f"{i}. {title}\n"
                results_text += f"   链接: {link}\n"
                results_text += f"   摘要: {snippet}\n\n"
            
            return results_text
        
        except requests.RequestException as e:
            logger.error(f"Bing搜索请求失败: {str(e)}")
            return f"搜索失败: {str(e)}"
        except Exception as e:
            logger.error(f"Bing搜索过程中发生错误: {str(e)}")
            return f"搜索错误: {str(e)}"