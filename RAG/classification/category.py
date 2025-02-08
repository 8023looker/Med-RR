import ujson
import json

nlp_seq_classification = {
    "factual_question": "询问具体且客观的事实或数据",
    "definition_question": "询问事物的定义或解释",
    "explanatory_question": "寻求对现象、过程或事件原因的解释",
    "descriptive_question": "要求对事物的特征、性质、特点等进行描述",
    "directive_question": "用来请求指导或建议",
    "opinion_question": "设计个人感受、态度或偏好",
    "comparative_question": "用于对比两个或多个事物之间的差异",
    "evaluative_question": "用于评估某个陈述或观点的正确性或质量",
    "hypothetical_question": "提出一种假设情景，并要求预测结果或反应",
    "procedural_question": "询问完成某项任务或活动的具体步骤",
    "referential_question": "通过引用特定文档、资源或其他信息来寻求答案",
    "verification_question": "确认或核实某些信息的真实性或准确性"
}

# EBM
evidence_based_medicine = {
    "prognosis": "预后",
    "therapy": "治疗",
    "etiology": "病因",
    "diagnosis": "诊断",
    "prevention": "预防",
    "cost": "成本",
    "others": "其他"
}