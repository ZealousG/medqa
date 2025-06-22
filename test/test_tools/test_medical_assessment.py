import pytest
import asyncio
from src.tools.medical_assessment_tool import MedicalAssessmentTool, MedicalAssessmentInput

class TestMedicalAssessmentTool:
    """医疗评估工具测试类"""
    
    @pytest.fixture
    def medical_assessment(self):
        """创建医疗评估工具实例"""
        return MedicalAssessmentTool()
    
    def test_tool_properties(self, medical_assessment):
        """测试工具属性"""
        assert medical_assessment.name == "医疗评估"
        assert "健康风险评估" in medical_assessment.description
        assert "疾病筛查" in medical_assessment.description
        assert medical_assessment.args_schema == MedicalAssessmentInput
    
    def test_input_schema(self, medical_assessment):
        """测试输入参数模式"""
        # 验证参数模式
        assert medical_assessment.args_schema == MedicalAssessmentInput
        
        # 验证参数字段
        fields = medical_assessment.args_schema.model_fields
        assert "assessment_type" in fields
        assert "patient_data" in fields
        assert fields["assessment_type"].annotation == str
        assert "Dict" in str(fields["patient_data"].annotation)
    
    def test_cardiovascular_risk_assessment(self, medical_assessment):
        """测试心血管风险评估"""
        # 正常情况测试
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
        
        result = medical_assessment._run("心血管风险", patient_data)
        assert "心血管疾病风险评估结果" in result
        assert "10年心血管疾病风险" in result
        assert "风险分类" in result
        assert "评估建议" in result
        
        # 高风险情况测试
        high_risk_data = {
            "age": 65,
            "gender": "male",
            "total_cholesterol": 280,
            "hdl_cholesterol": 35,
            "systolic_bp": 160,
            "is_smoker": True,
            "is_diabetic": True,
            "is_treated_bp": True
        }
        
        result = medical_assessment._run("心血管风险", high_risk_data)
        assert "心血管疾病风险评估结果" in result
        assert "高风险" in result or "中等风险" in result
    
    def test_diabetes_risk_assessment(self, medical_assessment):
        """测试糖尿病风险评估"""
        # 低风险情况
        low_risk_data = {
            "age": 35,
            "bmi": 22,
            "waist_circumference": 85,
            "physical_activity": True,
            "vegetables_fruits_berries": True,
            "hypertension_medication": False,
            "high_blood_glucose": False,
            "family_diabetes": "no"
        }
        
        result = medical_assessment._run("糖尿病风险", low_risk_data)
        assert "2型糖尿病风险评估结果" in result
        assert "FINDRISC得分" in result
        assert "风险等级" in result
        
        # 高风险情况
        high_risk_data = {
            "age": 60,
            "bmi": 32,
            "waist_circumference": 110,
            "physical_activity": False,
            "vegetables_fruits_berries": False,
            "hypertension_medication": True,
            "high_blood_glucose": True,
            "family_diabetes": "yes_immediate"
        }
        
        result = medical_assessment._run("糖尿病风险", high_risk_data)
        assert "2型糖尿病风险评估结果" in result
        assert "高风险" in result or "非常高风险" in result
    
    def test_depression_screening(self, medical_assessment):
        """测试抑郁症筛查"""
        # 无抑郁症状
        no_depression_data = {
            "phq9_answers": [0, 0, 0, 0, 0, 0, 0, 0, 0]
        }
        
        result = medical_assessment._run("抑郁症筛查", no_depression_data)
        assert "抑郁症筛查结果" in result
        assert "无抑郁症状" in result
        
        # 轻度抑郁
        mild_depression_data = {
            "phq9_answers": [1, 1, 1, 1, 1, 1, 1, 1, 0]
        }
        
        result = medical_assessment._run("抑郁症筛查", mild_depression_data)
        assert "抑郁症筛查结果" in result
        assert "轻度抑郁" in result
        
        # 重度抑郁
        severe_depression_data = {
            "phq9_answers": [3, 3, 3, 3, 3, 3, 3, 3, 3]
        }
        
        result = medical_assessment._run("抑郁症筛查", severe_depression_data)
        assert "抑郁症筛查结果" in result
        assert "重度抑郁" in result
        assert "自杀风险" in result
    
    def test_anxiety_screening(self, medical_assessment):
        """测试焦虑症筛查"""
        # 无焦虑症状
        no_anxiety_data = {
            "gad7_answers": [0, 0, 0, 0, 0, 0, 0]
        }
        
        result = medical_assessment._run("焦虑症筛查", no_anxiety_data)
        assert "焦虑症筛查结果" in result
        assert "无焦虑症状" in result
        
        # 中度焦虑
        moderate_anxiety_data = {
            "gad7_answers": [2, 2, 2, 2, 2, 2, 2]
        }
        
        result = medical_assessment._run("焦虑症筛查", moderate_anxiety_data)
        assert "焦虑症筛查结果" in result
        assert "中度焦虑" in result
    
    def test_invalid_assessment_type(self, medical_assessment):
        """测试无效的评估类型"""
        patient_data = {"age": 30}
        result = medical_assessment._run("无效类型", patient_data)
        assert "不支持的评估类型" in result
    
    def test_missing_required_fields(self, medical_assessment):
        """测试缺少必填字段"""
        # 心血管风险评估缺少字段
        incomplete_data = {
            "age": 45,
            "gender": "male"
            # 缺少其他必填字段
        }
        
        result = medical_assessment._run("心血管风险", incomplete_data)
        assert "缺少必要的患者数据字段" in result
        
        # 糖尿病风险评估缺少字段
        incomplete_diabetes_data = {
            "age": 45
            # 缺少其他必填字段
        }
        
        result = medical_assessment._run("糖尿病风险", incomplete_diabetes_data)
        assert "缺少必要的患者数据字段" in result
    
    def test_invalid_phq9_data(self, medical_assessment):
        """测试无效的PHQ-9数据"""
        # 缺少PHQ-9数据
        result = medical_assessment._run("抑郁症筛查", {})
        assert "缺少PHQ-9问卷回答数据" in result
        
        # PHQ-9数据长度不正确
        invalid_data = {
            "phq9_answers": [0, 1, 2]  # 只有3个答案，应该是9个
        }
        
        result = medical_assessment._run("抑郁症筛查", invalid_data)
        assert "PHQ-9问卷应包含9个问题的回答" in result
        
        # PHQ-9数据格式不正确
        invalid_format_data = {
            "phq9_answers": ["a", "b", "c", "d", "e", "f", "g", "h", "i"]  # 不是数字
        }
        
        result = medical_assessment._run("抑郁症筛查", invalid_format_data)
        assert "PHQ-9回答应为0-3的整数" in result
    
    def test_invalid_gad7_data(self, medical_assessment):
        """测试无效的GAD-7数据"""
        # 缺少GAD-7数据
        result = medical_assessment._run("焦虑症筛查", {})
        assert "缺少GAD-7问卷回答数据" in result
        
        # GAD-7数据长度不正确
        invalid_data = {
            "gad7_answers": [0, 1, 2]  # 只有3个答案，应该是7个
        }
        
        result = medical_assessment._run("焦虑症筛查", invalid_data)
        assert "GAD-7问卷应包含7个问题的回答" in result
    
    def test_data_type_conversion(self, medical_assessment):
        """测试数据类型转换"""
        # 字符串数字应该能正确转换为数字
        patient_data = {
            "age": "45",
            "gender": "male",
            "total_cholesterol": "200",
            "hdl_cholesterol": "50",
            "systolic_bp": "130",
            "is_smoker": "false",
            "is_diabetic": "false",
            "is_treated_bp": "false"
        }
        
        result = medical_assessment._run("心血管风险", patient_data)
        assert "心血管疾病风险评估结果" in result
        assert "计算错误" not in result
    
    def test_edge_cases(self, medical_assessment):
        """测试边界情况"""
        # 极低年龄
        young_patient = {
            "age": 20,
            "gender": "male",
            "total_cholesterol": 200,
            "hdl_cholesterol": 50,
            "systolic_bp": 120,
            "is_smoker": False,
            "is_diabetic": False,
            "is_treated_bp": False
        }
        
        result = medical_assessment._run("心血管风险", young_patient)
        assert "心血管疾病风险评估结果" in result
        
        # 极高年龄
        old_patient = {
            "age": 80,
            "gender": "female",
            "total_cholesterol": 200,
            "hdl_cholesterol": 50,
            "systolic_bp": 120,
            "is_smoker": False,
            "is_diabetic": False,
            "is_treated_bp": False
        }
        
        result = medical_assessment._run("心血管风险", old_patient)
        assert "心血管疾病风险评估结果" in result
    
    def test_async_execution(self, medical_assessment):
        """测试异步执行"""
        async def test_async():
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
            
            result = await medical_assessment._arun("心血管风险", patient_data)
            assert "心血管疾病风险评估结果" in result
        
        asyncio.run(test_async())
    
    def test_error_handling(self, medical_assessment):
        """测试错误处理"""
        # 无效的数据类型
        invalid_data = {
            "age": "invalid_age",
            "gender": "male",
            "total_cholesterol": 200,
            "hdl_cholesterol": 50,
            "systolic_bp": 130,
            "is_smoker": False,
            "is_diabetic": False,
            "is_treated_bp": False
        }
        
        result = medical_assessment._run("心血管风险", invalid_data)
        assert "评估过程中发生错误" in result or "计算错误" in result
    
    def test_comprehensive_assessment_workflow(self, medical_assessment):
        """测试完整的评估工作流程"""
        # 测试所有评估类型
        assessment_types = ["心血管风险", "糖尿病风险", "抑郁症筛查", "焦虑症筛查"]
        
        for assessment_type in assessment_types:
            if assessment_type == "心血管风险":
                patient_data = {
                    "age": 50,
                    "gender": "female",
                    "total_cholesterol": 220,
                    "hdl_cholesterol": 55,
                    "systolic_bp": 140,
                    "is_smoker": False,
                    "is_diabetic": False,
                    "is_treated_bp": False
                }
            elif assessment_type == "糖尿病风险":
                patient_data = {
                    "age": 50,
                    "bmi": 28,
                    "waist_circumference": 95,
                    "physical_activity": True,
                    "vegetables_fruits_berries": True,
                    "hypertension_medication": False,
                    "high_blood_glucose": False,
                    "family_diabetes": "yes_distant"
                }
            elif assessment_type == "抑郁症筛查":
                patient_data = {
                    "phq9_answers": [1, 1, 0, 1, 0, 1, 0, 1, 0]
                }
            elif assessment_type == "焦虑症筛查":
                patient_data = {
                    "gad7_answers": [1, 0, 1, 0, 1, 0, 1]
                }
            
            result = medical_assessment._run(assessment_type, patient_data)
            assert "评估结果" in result or "筛查结果" in result
            assert "建议" in result or "干预措施" in result

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 