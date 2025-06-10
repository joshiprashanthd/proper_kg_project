from pydantic import BaseModel
from src.utils import OpenAIModel

class ReasoningPathLabel(BaseModel):
    label: str
    justification: str
    reasoning_path: str

class UsefulnessResponseFormat(BaseModel):
    reasoning_paths_with_labels: list[ReasoningPathLabel]

class ReasoningPathUsefulness:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model = OpenAIModel(model_name, 'cuda')
        self.prompt_template = """You are a medical expert. Your goal is to evaluate the usefulness of the following reasoning paths given question and answer pair.

A reasoning path is structured as follows:
(entity 1) -- relation 1 --> (entity 2) -- relation 2 --> (entity 3) -- ... -- relation n --> (entity n+1)

Usefulness of a reasoning path can be determined by following criteria:
- The entities in the path should have a clear semantic connection to the core concepts in both question and answer. 
- If a path introduces completely unrelated concepts, it's likely unhelpful.
- The relationships in the path should ideally lead from concepts present in the question towards the concepts present in the answer. The flow of reasoning should make intuitive sense in this direction.
- Each step in the reasoning path (entity -- relation --> entity) should represent a meaningful and logical connection within the mental health domain. Avoid paths with obscure or nonsensical relationships.
- Ensure all entities and relations within the path belong to the mental health domain or closely related areas. Paths that veer into unrelated medical or general knowledge domains might not be helpful for your specific goal.
- While not a strict filter, very long reasoning paths might introduce too many intermediate and potentially irrelevant concepts. Shorter, more direct paths are often more effective for explanation. You might consider setting a reasonable upper limit on path length or prioritizing shorter paths initially.

Your task is to evaluate the usefulness of the reasoning paths based on these criteria by assigning a label to each path:
1. Useful
2. Not Useful

Read the question and answer pair carefully, then evaluate the reasoning paths one by one. For each reasoning path, provide a label and a brief justification for your evaluation.

Question: {question}
Answer: {answer}
Reasoning Paths:
{reasoning_paths}

Output:
"""
    def run(self, question: str, answer: str, reasoning_paths: list[str]):
        response = self.model.generate_text(
            self.prompt_template.format(question=question, answer=answer, reasoning_paths="\n".join(reasoning_paths)),
            structured_format=UsefulnessResponseFormat,
            max_tokens=8192
        )
        return response.reasoning_paths_with_labels