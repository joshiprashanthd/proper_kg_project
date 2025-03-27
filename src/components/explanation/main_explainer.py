from .base_explainer import BaseExplainer
from src.utils import OpenAIModel

class MainExplainer(BaseExplainer):
    def __init__(self):
        self.prompt_template = """You are a medical expert. Your goal is to generate an explanation only using given reasoning paths such that the explanation helps in arriving to the correct answer to the given question.

The reasoning path is of the format: (entity 1) -> relation 1 -> (entity 2) -> relation 2 -> .... -> (entity n)
Entities are enclosed in parentheses.

Task Instructions:
- Read each reasoning path carefully and try to interpret the relationship between entities.
- Generate a comprehensive explanation that uses multiple reasoning paths as reference.
- The explanation should give insights obtained from the reasoning paths that could help in answering the question.
- Always add conclusion section where you summarize all the insights gathered from reasoning paths.
- Always provide explanations and reasoning that leads to given correct answer.
- Always cite reasoning path(s) that you used to explain or extract insights from.

NOTE: There could be a case where you might not get any reasoning paths. In such cases, you can generate explanation based on your own knowledge. After generation explanation based on your knowledge, you can provide reasoning paths that you think could have been used to arrive at the answer.

Question: {question}
Correct Answer: {answer}

{qna_context_prefix}
{qna_context}

{context_prefix}
{context}

Explanation: """
        self.model = OpenAIModel("gpt-4o-mini", 'cuda')

    def run(self, question: str, answer: str, qna_context_prefix = "", qna_context="", context_prefix="", context=""):
        prompt = self.prompt_template.format(question=question, answer=answer, qna_context_prefix=qna_context_prefix, qna_context=qna_context, context_prefix=context_prefix, context=context)
        return self.model.generate_text(prompt, max_tokens=1024)
