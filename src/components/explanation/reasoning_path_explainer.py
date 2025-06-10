from src.utils import OpenAIModel

class ReasoningPathExplainer:
    def __init__(self):
        self.prompt_template = """You are a medical expert. Your goal is to generate an explanation that could answer the question given in the text.

Task Instructions:
1.  Mentally analyze the provided reasoning paths to understand the medical connections they represent in relation to the question and the correct answer. *Ignore any provided paths that do not strictly follow the format `(entity) -- [relation] -> (entity) -- ...` as they are malformed.*
2.  Structure your explanation by presenting the key steps or connections (derived from the insights of the most relevant valid provided paths) that logically lead from the question's subject to the correct answer.
3.  Integrate the insights from the relevant valid provided paths into your explanation as if they are your own reasoning steps. You may present a path explicitly as a step in your thinking process or describe the connection it represents.
4.  DO NOT use phrases like "provided reasoning path", "given path", "from Reasoning Path X", "according to the path", or any language suggesting the paths were external inputs you received. Your language should imply these reasoning steps are part of your own medical deduction.
5.  Explain how these reasoning steps and the medical connections they highlight logically support the given correct answer.
6.  Combine insights from multiple relevant valid paths to form a cohesive and comprehensive explanation of the reasoning process.
7.  If the provided valid paths are insufficient to fully explain the correct answer, your explanation should still be based only on the insights inferable from those paths, presenting them as your reasoning steps without introducing outside knowledge or fabricating connections not present in the paths.
8.  Always include a 'Conclusion' section at the end where you summarize the key takeaways from the reasoning steps presented in your explanation.

NOTE: If no reasoning paths are provided or the list of reasoning paths provided is empty or contains only malformed paths, generate an explanation based only on your general medical knowledge relevant to the question and answer. In this specific case (and only this case), your explanation will not involve presenting specific reasoning path formats, but you should still provide a 'Suggested Reasoning Paths' section at the end listing potential reasoning paths (in the specified format) that a medical expert could use to connect the question entities to the answer entities.


<query>
{query}
</query>

<context>
{context}
</context>

Explanation:
"""
        self.model = OpenAIModel("gpt-4o-mini", 'cuda')

    def run(self, query: str, context: str):
        prompt = self.prompt_template.format(query=query, context=context)
        return self.model.generate_text(prompt, max_tokens=1024)
