import pytest
import math
from src.tools.calculator_tool import CalculatorTool, CalculatorInput

class TestCalculatorTool:
    """计算工具测试类"""
    
    @pytest.fixture
    def calculator(self):
        """创建计算工具实例"""
        return CalculatorTool()
    
    def test_basic_arithmetic(self, calculator):
        """测试基本算术运算"""
        # 加法
        result = calculator._run("2 + 3")
        assert "计算结果: 5" in result
        
        # 减法
        result = calculator._run("10 - 4")
        assert "计算结果: 6" in result
        
        # 乘法
        result = calculator._run("6 * 7")
        assert "计算结果: 42" in result
        
        # 除法
        result = calculator._run("15 / 3")
        assert "计算结果: 5.0" in result
        
        # 幂运算
        result = calculator._run("2 ^ 3")
        assert "计算结果: 8" in result
        
        # 复杂表达式
        result = calculator._run("(2 + 3) * 4 - 1")
        assert "计算结果: 19" in result
    
    def test_math_functions(self, calculator):
        """测试数学函数"""
        # 平方根
        result = calculator._run("sqrt(16)")
        assert "计算结果: 4.0" in result
        
        # 绝对值
        result = calculator._run("abs(-5)")
        assert "计算结果: 5" in result
        
        # 四舍五入
        result = calculator._run("round(3.14159, 2)")
        assert "计算结果: 3.14" in result
        
        # 三角函数
        result = calculator._run("sin(0)")
        assert "计算结果: 0.0" in result
        
        result = calculator._run("cos(0)")
        assert "计算结果: 1.0" in result
        
        # 对数
        result = calculator._run("log10(100)")
        assert "计算结果: 2.0" in result
    
    def test_medical_dose_calculation(self, calculator):
        """测试医疗剂量计算"""
        # 标准剂量计算
        result = calculator._run("10 mg/kg * 70 kg")
        assert "剂量计算结果: 700.0 mg" in result
        
        # 小数剂量计算
        result = calculator._run("0.5 mg/kg * 50 kg")
        assert "剂量计算结果: 25.0 mg" in result
        
        # 使用中文乘号
        result = calculator._run("5 mg/kg × 60 kg")
        assert "剂量计算结果: 300.0 mg" in result
    
    def test_bmi_calculation(self, calculator):
        """测试BMI计算"""
        # 标准BMI计算
        result = calculator._run("BMI: 70 kg / (1.75 m)^2")
        assert "BMI计算结果: 22.86" in result
        assert "正常范围" in result
        
        # 体重过轻 - 修复精度问题
        result = calculator._run("BMI: 45 kg / (1.70 m)²")
        assert "BMI计算结果: 15.57" in result  # 修正为实际计算结果
        assert "体重过轻" in result
        
        # 超重
        result = calculator._run("BMI: 80 kg / (1.65 m)^2")
        assert "BMI计算结果: 29.38" in result
        assert "轻度肥胖" in result
        
        # 重度肥胖
        result = calculator._run("BMI: 120 kg / (1.60 m)²")
        assert "BMI计算结果: 46.87" in result
        assert "重度肥胖" in result
    
    def test_body_surface_area_calculation(self, calculator):
        """测试体表面积计算"""
        # 标准体表面积计算
        result = calculator._run("体表面积: 70 kg, 170 cm")
        assert "体表面积计算结果:" in result
        assert "m²" in result
        
        # 儿童体表面积计算
        result = calculator._run("体表面积: 25 kg, 120 cm")
        assert "体表面积计算结果:" in result
        assert "m²" in result
    
    def test_error_handling(self, calculator):
        """测试错误处理"""
        # 无效表达式
        result = calculator._run("invalid expression")
        assert "计算错误:" in result
        
        # 除零错误
        result = calculator._run("10 / 0")
        assert "计算错误:" in result
        
        # 未定义的函数
        result = calculator._run("undefined_function(10)")
        assert "计算错误:" in result
    
    def test_input_schema(self, calculator):
        """测试输入参数模式"""
        # 验证参数模式
        assert calculator.args_schema == CalculatorInput
        
        # 验证参数字段 - 修复 Pydantic V2 API
        fields = calculator.args_schema.model_fields
        assert "expression" in fields
        assert fields["expression"].annotation == str
    
    def test_async_execution(self, calculator):
        """测试异步执行"""
        import asyncio
        
        async def test_async():
            result = await calculator._arun("2 + 2")
            assert "计算结果: 4" in result
        
        asyncio.run(test_async())
    
    def test_bmi_categories(self, calculator):
        """测试BMI分类功能"""
        # 测试所有BMI分类
        test_cases = [
            (17.0, "体重过轻"),
            (22.0, "正常范围"),
            (26.0, "超重"),
            (29.0, "轻度肥胖"),
            (32.0, "中度肥胖"),
            (40.0, "重度肥胖")
        ]
        
        for bmi, expected_category in test_cases:
            category = calculator._get_bmi_category(bmi)
            assert category == expected_category
    
    def test_special_characters(self, calculator):
        """测试特殊字符处理"""
        # 中文乘号
        result = calculator._run("3 × 4")
        assert "计算结果: 12" in result
        
        # 中文除号
        result = calculator._run("12 ÷ 3")
        assert "计算结果: 4.0" in result
        
        # 幂运算符号
        result = calculator._run("2 ^ 3")
        assert "计算结果: 8" in result
        
        # 带空格的表达式
        result = calculator._run(" 2 + 3 ")
        assert "计算结果: 5" in result
    
    def test_complex_medical_calculations(self, calculator):
        """测试复杂医疗计算"""
        # 复合计算：BMI + 剂量
        result = calculator._run("(70 / (1.75 ^ 2)) + (0.1 * 70)")
        assert "计算结果:" in result
        
        # 体表面积相关计算
        result = calculator._run("sqrt((70 * 170) / 3600)")
        assert "计算结果:" in result
    
    def test_tool_properties(self, calculator):
        """测试工具属性"""
        assert calculator.name == "计算工具"
        assert "执行数学计算" in calculator.description
        assert "医疗剂量计算" in calculator.description
        assert "BMI计算" in calculator.description

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 