import ujson
import json


cot_english_prompt_template = """
Given the provided [context] and the specific [question], as well as the relevant documents retrieved through a query, please provide an answer that includes your thought process. Specifically:

1. **Analyze the Question**: Carefully analyze the [question] to understand what information is being sought.
2. **Review Provided Context**: Examine the [context] for any background information that can help frame the answer.
3. **Consult Retrieved Documents**: Go through the document snippets retrieved by the query to identify sections that are directly related to the [question].
4. **Identify Key Information**: Highlight the key points from the retrieved documents that address the question's requirements.
5. **Construct Thought Process**: Explain how you used the information from the [context] and the retrieved documents to form your understanding and construct your answer.
6. **Provide Answer**: Finally, give a clear and concise answer to the [question], supported by the analysis of the retrieved documents.

Please present your response in a way that clearly shows your reasoning and the sources of information you relied on.
"""

cot_chinese_prompt_template = """
针对提供的[背景信息]和具体的[问题]，以及通过查询检索到的相关文档，请给出一个包含你思考过程的答案。具体来说：

1. **分析问题**：仔细分析[问题]，以明确需要寻找的信息。
2. **审查提供的背景信息**：检查[背景信息]，以获取可以帮助构建答案的任何背景资料。
3. **查阅检索到的文档**：浏览通过查询检索到的文档片段，找出与[问题]直接相关的部分。
4. **确定关键信息**：从检索到的文档中突出显示能够解答问题的关键点。
5. **构建思考过程**：解释你是如何利用[背景信息]和检索到的文档中的信息来形成理解和构建答案的。
6. **提供答案**：最后，给出一个清晰简洁的答案回应[问题]，并且这个答案应得到所分析的检索文档的支持。

请以一种能清晰展示你的推理过程和依赖的信息来源的方式呈现你的回答。
"""

COT_PROMPTS = {
    "en": (
        "Given the provided [Context] and the specific [Question], as well as the [Relevant Documents Retrieved through a Query], please provide an answer that includes your thought process. Specifically:\n\n"
        "1. **Analyze the Question**: Carefully analyze the [Question] to understand what information is being sought.\n"
        "2. **Review Provided Context**: Examine the [Context] for any background information that can help frame the answer.\n"
        "3. **Consult [Relevant Documents Retrieved through a Query]**: Go through the snippets of [Relevant Documents Retrieved through the Query] to identify sections that are directly related to the [Question].\n"
        "4. **Identify Key Information**: Highlight the key points from the [Relevant Documents Retrieved through a Query] that address the question's requirements.\n"
        "5. **Construct Thought Process**: Explain how you used the information from the [Context] and the retrieved documents to form your understanding and construct your answer.\n"
        "6. **Provide Answer**: Finally, give a clear and concise answer to the [Question], supported by the analysis of the [Relevant Documents Retrieved through a Query].\n\n"
        "Please present your response in a way that clearly shows your reasoning and the sources of information you relied on."
    ),
    "zh": (
        "针对提供的[背景信息]和具体的[问题]，以及通过查询[检索到的相关文档]，请给出一个包含你思考过程的答案。具体来说：\n\n"
        "1. **分析问题**：仔细分析[问题]，以明确需要寻找的信息。\n"
        "2. **审查提供的背景信息**：检查[背景信息]，以获取可以帮助构建答案的任何背景资料。\n"
        "3. **查阅[检索到的相关文档]**：浏览通过查询[检索到的相关文档]的片段，找出与[问题]直接相关的部分。\n"
        "4. **确定关键信息**：从[检索到的相关文档]中突出显示能够解答问题的关键点。\n"
        "5. **构建思考过程**：解释你是如何利用[背景信息]和[检索到的相关文档]中的信息来形成理解和构建答案的。\n"
        "6. **提供答案**：最后，给出一个清晰简洁的答案回应[问题]，并且这个答案应得到所分析的[检索到的相关文档]的支持。\n\n"
        "请以一种能清晰展示你的推理过程和依赖的信息来源的方式呈现你的回答。"
    )
}

REPLACE_DICT = {
    "context": {
        "en": "[Context]",
        "zh": "[背景信息]"
    },
    "question": {
        "en": "[Question]",
        "zh": "[问题]"
    },
    "retrieved_documents": {
        "en": "[Relevant Documents Retrieved through a Query]",
        "zh": "[检索到的相关文档]"
    }
}
