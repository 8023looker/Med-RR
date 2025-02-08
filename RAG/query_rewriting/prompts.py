import ujson
import json

classification_dict = {
    "EBM_classification": { # 询证分类
        "zh": "你是一个医疗领域的句子标注专家，现有以下七种临床问题：预后，治疗，病因，诊断，预防，成本，其他。根据句子的目的和结构对以下文本片段进行分类，直接给出分类名，不要有其他表述:",
        "en": "You are an expert in sentence annotation within the medical field. There are seven categories of clinical questions: Prognosis, Therapy, Etiology, Diagnosis, Prevention, Cost, and Other. Please classify the following text fragment based on their purpose and structure by providing only the category name without additional commentary:"
    },
    "question_classification": { # 问题分类
        "zh": "你是一个句子标注专家，现有以下 13 种问题分类：事实型，定义型，解释型，描述型，指示型，意见型，比较型，评价型，假设型，程序型，参考型，验证型，其他类型。根据句子的目的和结构对以下文本片段进行分类，直接给出分类名，不要有其他表述:",
        "en": "You are an expert in sentence annotation. Given the following 13 categories of question types: Factual, Definitional, Explanatory, Descriptive, Directive, Opinion, Comparative, Evaluative, Hypothetical, Procedural, Referential, Verification, and Other.  Please classify the following text fragment based on their purpose and structure by providing only the category name without additional commentary:"
    }
}

EBM_rewriting_dict = {
    "prognosis": { # 预后
        "zh": "请明确指出疾病的名称，并询问其长期结果或患者结局，如生存率、复发率等。",
        "en": "Please specify the disease or condition and ask about long-term outcomes such as survival rates, recovery chances, or disease progression."
    },
    "therapy": { # 治疗
        "zh": "请具体说明疾病或症状，以及所考虑的治疗方法，询问其疗效、安全性或与其他疗法的比较效果。",
        "en": "Please specify the disease or symptom along with the therapy being considered, and inquire about its effectiveness, safety, or comparison with other therapies."
    },
    "etiology": { # 病因
        "zh": "请描述疾病或健康问题，并询问可能导致它的原因，包括风险因素、病原体或遗传背景。",
        "en": "Please describe the health issue and ask about potential causes, including risk factors, pathogens, or genetic background."
    },
    "diagnosis": { # 诊断
        "zh": "请指明需要诊断的病症或状况，并询问特定检测方法的准确性、灵敏度或特异性。",
        "en": "Please specify the condition you need to diagnose and ask about the accuracy, sensitivity, or specificity of specific diagnostic tests."
    },
    "prevention": { # 预防
        "zh": "请指定要预防的疾病或健康问题，并询问预防措施的效果或建议。",
        "en": "Please specify the disease or health issue and ask about the effectiveness of preventive measures or recommendations."
    },
    "cost": { # 成本
        "zh": "请明确指出医疗干预或服务，并询问其成本效益分析，包括直接和间接成本，以及成本-效果比。",
        "en": "Please specify the medical intervention or service and ask about cost-effectiveness analyses, including direct and indirect costs and cost-effectiveness ratios."
    },
    "others": { # 其他
        "zh": "请详细说明涉及的问题领域，例如伦理、法律或患者教育，并询问相关信息或指南。",
        "en": "Please detail the area of interest, such as ethics, legal issues, or patient education, and ask for relevant information or guidelines."
    }
}