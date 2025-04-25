from .base_ner import BaseNER
from pydantic import BaseModel

class NERResponseFormat(BaseModel):
            entities: list[str]

class OpenAINER(BaseNER):
    def __init__(self, model: BaseModel =  None):
        super().__init__(model)
        self.prompt_template = """You are a medical expert performing named entity recognition on mental health text. Your goal is to extract relevant medical and mental health entities.

Please identify:
1. Medical conditions and diseases
2. Medications and treatments
3. Symptoms
4. Medical procedures
5. Anatomical structures
6. Medical terminology

For each category, identify short terms that are nouns or noun phrases, typically 1 to 3 words long. Do NOT output long phrases or full sentences as entities.

Handle Edge Cases:
- Focus on entities relevant to the main subject of the text (e.g., the patient). Do not include entities mentioned only in a hypothetical context or as family history.
- Do not include medical entities that are explicitly negated (e.g., "no pain", "ruled out diabetes").
- If a concept is mentioned multiple times, extract it only once.

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