import ujson
import json

retrieval_category_prompt = { # 询证分类 question-answer mapping
    "zh": "你是一个医疗领域的句子标注专家，现有以下 16 种句子类型：论证，定义，描述，解释，目的，叙述，过程，指令，命令，问题解决，比较，评价，分类，条件，预测，因果。根据句子的目的和结构对以下文本片段进行分类，给出其所属于各个类别的归一化概率分布，参照 [x1, x2, x3, ..., x16] 的形式，各类别的概率之和为1，不要有其他表述:",
    "en": "You are an expert in sentence annotation within the medical field. There are 16 categories of documents: Argumentation, Definition, Description, Explanation, Purpose, Narration, Process, Instruction, Command, Problem-Solving, Comparison, Evaluation, Classification, Condition, Prediction, Cause-and-Effect. Please classify the following text fragment based on their purpose and structure by providing the probability distribution of its belonging to each category, in the format of [x1, x2, x3, ..., x16], where the sum of probabilities across all categories equals 1, without additional commentary:"
}

evidence_level_prompt = { # 证据级别 question-answer mapping
    "zh": "您是医学领域证据质量标注的专家。文件共有9个质量等级：元分析、系统综述、循证实践指南、随机对照试验、非随机对照试验、队列研究、病例系列或研究、个案报告、专家意见。根据句子的目的和结构对以下文本片段进行等级划分，仅给出等级名称，不要有其他表述:",
    "en": "You are an expert in evidence quality annotation within the medical field. There are 9 quality levels of documents: Meta-Analyses, Systematic Reviews, Evidence-Based Practice Guidelines, Randomized Controlled Trials, Non-Randomized Controlled Trials, Cohort Studies, Case Series or Studies, Individual Case Reports, Expert Opinion. Please classify the following text segment based on its purpose and structure, providing only the names of the levels, without any additional description:"
}

document_classification = {
    "argumentation": "提出了一个观点或论点，并可能提供支持的论据",
    "definition": "提供了一个术语或概念的明确定义",
    "description": "描述物体或事件的特征或属性",
    "explanation": "解释了某个概念、过程或原因",
    "purpose": "解释了某个行为或事件的目的或意图",
    "narration": "讲述了一个事件、经历或故事",
    "process": "描述了一个过程或一系列步骤",
    "instruction": "提供了执行任务或操作的步骤或指导",
    "command": "表达了一个请求或命令，要求听者采取行动",
    "problem-solving": "提出了解决特定问题的方法或策略",
    "comparison": "比较了两个或多个事物的相似性或差异",
    "evaluation": "表达了对某个主题或行为的评价或判断",
    "classification": "将事物或概念归入特定的类别或类别体系",
    "condition": "描述了某个事件发生的条件或假设",
    "prediction": "对未来的事件或趋势进行预测",
    "cause-and-effect": "描述了事件之间的因果关系"
}

query_document_projection = {
    "factual": ["argumentation", "definition", "description", "explanation", "purpose", "narration"],
    "procedural": ["purpose", "instruction", "command", "problem-solving"],
    "comparative": ["comparison", "evaluation", "classification"],
    "hypothetical": ["condition", "prediction", "cause-and-effect"]
}

document_classification_list = [
    "argumentation",
    "definition",
    "description",
    "explanation",
    "purpose",
    "narration",
    "process",
    "instruction",
    "command",
    "problem-solving",
    "comparison",
    "evaluation",
    "classification",
    "condition",
    "prediction",
    "cause-and-effect"
]

evidence_level = {
    "meta-analyses": 9, 
    "systematic reviews": 8, 
    "evidence-based practice guidelines": 7, 
    "randomized controlled trials": 6, 
    "non-randomized controlled trials": 5, 
    "cohort studies": 4, 
    "case series or studies": 3,
    "individual case reports": 2, 
    "expert opinion": 1
}