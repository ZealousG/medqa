"""
LangChain 工具使用示例
展示如何使用重写后的医疗工具
"""

from src.tools.medical_assessment_tool import MedicalAssessmentTool
from src.tools.medical_reference_tool import MedicalReferenceTool
from src.tools.calculator_tool import CalculatorTool
from src.tools.search_tool import SearchTool

def main():
    """主函数，展示各种工具的使用"""
    
    print("=== LangChain 医疗工具使用示例 ===\n")
    
    # 1. 医疗评估工具示例
    print("1. 医疗评估工具示例:")
    assessment_tool = MedicalAssessmentTool()
    
    # 心血管风险评估
    patient_data = {
        "age": 45,
        "gender": "male",
        "total_cholesterol": 200,
        "hdl_cholesterol": 50,
        "systolic_bp": 130,
        "is_smoker": False,
        "is_diabetic": False,
        "is_treated_bp": False
    }
    
    result = assessment_tool._run("心血管风险", patient_data)
    print("心血管风险评估结果:")
    print(result[:200] + "..." if len(result) > 200 else result)
    print()
    
    # 2. 医疗参考工具示例
    print("2. 医疗参考工具示例:")
    reference_tool = MedicalReferenceTool()
    
    result = reference_tool._run("诊断标准", "糖尿病")
    print("糖尿病诊断标准查询结果:")
    print(result[:200] + "..." if len(result) > 200 else result)
    print()
    
    # 3. 计算工具示例
    print("3. 计算工具示例:")
    calculator = CalculatorTool()
    
    result = calculator._run("BMI: 70 kg / (1.75 m)^2")
    print("BMI计算结果:")
    print(result)
    print()
    
    # 4. 搜索工具示例（需要配置API密钥）
    print("4. 搜索工具示例:")
    search_tool = SearchTool()  # 未配置API密钥
    
    result = search_tool._run("糖尿病治疗", 3)
    print("搜索工具结果（未配置API）:")
    print(result)
    print()
    
    print("=== 示例完成 ===")

if __name__ == "__main__":
    main() 