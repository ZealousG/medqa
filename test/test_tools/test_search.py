import pytest
from src.tools.search_tool import SearchTool, SearchInput

class DummyResponse:
    def __init__(self, json_data, status_code=200):
        self._json = json_data
        self.status_code = status_code
    def json(self):
        return self._json
    def raise_for_status(self):
        if self.status_code != 200:
            raise Exception("HTTP error")

@pytest.fixture
def tool():
    return SearchTool(search_api_key="fake", search_engine_id="fake")

def test_tool_properties(tool):
    assert tool.name == "搜索工具"
    assert "搜索" in tool.description
    assert tool.args_schema == SearchInput

def test_no_api_key():
    t = SearchTool()
    result = t._run("test", 1)
    assert "未配置API密钥" in result

def test_no_results(monkeypatch, tool):
    def fake_get(*a, **k):
        return DummyResponse({})
    monkeypatch.setattr("requests.get", fake_get)
    result = tool._run("test", 1)
    assert "没有找到相关结果" in result

def test_success(monkeypatch, tool):
    def fake_get(*a, **k):
        return DummyResponse({"items": [{"title": "t1", "link": "l1", "snippet": "s1"}]})
    monkeypatch.setattr("requests.get", fake_get)
    result = tool._run("test", 1)
    assert "1. t1" in result
    assert "链接: l1" in result
    assert "摘要: s1" in result

def test_http_error(monkeypatch, tool):
    def fake_get(*a, **k):
        raise Exception("HTTP error")
    monkeypatch.setattr("requests.get", fake_get)
    result = tool._run("test", 1)
    assert "搜索错误" in result

def test_async(monkeypatch, tool):
    def fake_get(*a, **k):
        return DummyResponse({"items": [{"title": "t1", "link": "l1", "snippet": "s1"}]})
    monkeypatch.setattr("requests.get", fake_get)
    import asyncio
    async def run_async():
        result = await tool._arun("test", 1)
        assert "1. t1" in result
    asyncio.run(run_async())

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 