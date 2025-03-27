from .base_ner import BaseNER
from src.utils import OpenAIModel
from pydantic import BaseModel

class NERResponseFormat(BaseModel):
            entities: list[str]

class OpenAINER(BaseNER):
    def __init__(self):
        self.model = OpenAIModel("gpt-4o-mini", 'cuda')
        self.prompt_template = """You are a medical expert. Your goal is to perform named entity recognition on the following text. Try to find the medical and mental health related entities.

Please identify:
1. Medical conditions and diseases
2. Medications and treatments
3. Symptoms
4. Medical procedures
5. Anatomical structures
6. Medical terminology

If the text is ambiguous, unclear, or contains abbreviations, try to resolve them based on medical context.
If you're uncertain about an entity, include it and note your uncertainty.
If the text is empty or contains no medical entities, return an empty list.

Text:
{text}

Return only a list of identified entities without explanations.
"""

    def run(self, text: str):
        response = self.model.generate_text(self.prompt_template.format(text=text), structured_format=NERResponseFormat)
        return response.entities
    
    def run_mult(self, texts, type=None):
        responses = []
        for text in texts:
            response = self.model.generate_text(self.prompt_template.format(text=text), structured_format=NERResponseFormat)
            responses.append(response.entities)
        return responses