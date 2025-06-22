import pytest
from src.tools.medical_reference_tool import MedicalReferenceTool, MedicalReferenceInput

class TestMedicalReferenceTool:
    @pytest.fixture
    def tool(self):
        return MedicalReferenceTool()

    def test_tool_properties(self, tool):
        assert tool.name == "医疗参考"
        assert "参考" in tool.description
        assert tool.args_schema == MedicalReferenceInput

    def test_exact_match(self, tool):
        result = tool._run("诊断标准", "糖尿病")
        assert "诊断标准 - 糖尿病" in result
        assert "空腹血糖" in result

    def test_fuzzy_match(self, tool):
        result = tool._run("诊断标准", "糖")
        assert "找到与 '糖' 相关的" in result
        assert "糖尿病" in result

    def test_not_found(self, tool):
        result = tool._run("诊断标准", "不存在的疾病")
        assert "未找到与 '不存在的疾病' 相关的诊断标准信息" in result

    def test_invalid_type(self, tool):
        result = tool._run("未知类型", "糖尿病")
        assert "错误：不支持的查询类型" in result

    def test_async(self, tool):
        import asyncio
        async def run_async():
            result = await tool._arun("诊断标准", "糖尿病")
            assert "诊断标准 - 糖尿病" in result
        asyncio.run(run_async())

    def test_load_data_from_file(self, tool, tmp_path):
        # 构造一个临时json文件
        data = {"诊断标准": {"测试病": "测试内容"}}
        file_path = tmp_path / "ref.json"
        file_path.write_text(str(data).replace("'", '"'), encoding="utf-8")
        assert tool.load_data_from_file(str(file_path))
        assert tool.reference_data["诊断标准"]["测试病"] == "测试内容"

    def test_edge_case_empty_term(self, tool):
        result = tool._run("诊断标准", "")
        assert "诊断标准" in result or "未找到" in result

    def test_edge_case_empty_type(self, tool):
        result = tool._run("", "糖尿病")
        assert "错误：不支持的查询类型" in result

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 