from .base_ner import BaseNER
from pydantic import BaseModel
from src.utils import logger, OpenAIModel

class NERResponseFormat(BaseModel):
    question_entities: list[str]
    context_entities: list[str]
    answer_entities: list[str]

class OpenAINER(BaseNER):
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model = OpenAIModel(model_name, 'cuda')
        self.prompt_template = """Drawing extensively upon your knowledge of the UMLS Metathesaurus and its Semantic Network, perform medical named entity recognition for the mental health domain. Your goal is to identify and extract relevant medical and mental health entities based on UMLS concepts.

You should identify entities belonging to categories broadly aligned with UMLS Semantic Types, such as:
1. Medical conditions and diseases and disorders.
2. Medications and treatments
3. Symptoms
4. Medical procedures
5. Anatomical structures
6. Medical terminology (including key concepts, constructs, and psychosocial factors relevant to mental health research and practice, such as 'social support', 'quality of life', 'stigma', 'coping mechanisms', etc.)

Task Instruction:
- Extract entities from the question, context, and answer separately.
- For each entity identified using your UMLS knowledge, provide its most representative short term (noun or noun phrases) as found or implied in the text.
- DO NOT output long phrases or full sentences as entities.
- Always clean up the entities of any extra spaces or punctuation, standardizing terms using common UMLS synonyms or preferred terms where appropriate and consistent with the text.
- If an identified entity represents a broader concept that has narrower terms (subentities) in UMLS hierarchies, extract the subentities as well if they are mentioned or clearly implied in the text. For example, if "diabetes mellitus" is mentioned, also identify "diabetes" if it's present or a standard subentity/synonym.
- If a concept is mentioned multiple times, extract it only once using its most representative term.
- If the text contains few, ambiguous, or subtly implied medical entities, use your deep medical knowledge to identify relevant concepts and their common synonyms or related terms that fit the context.
- If the text for a section (Question, Context, or Answer) is truly empty and contains no text, return an empty list for that section.
- Also use your deep medical knowledge to think of other relevant entities that could potentially help in understanding and answering the question, even if they are not explicitly mentioned in the text.
- DO NOT BE FRUGAL IN EXTRACTING ENTITIES.

Perform NER on the following text:
Question: {question}
Context: {context}
Answer: {answer}
"""

    def run(self, question: str, context: str, answer: str):
        response = self.model.generate_text(self.prompt_template.format(question=question, context=context, answer=answer), structured_format=NERResponseFormat)
        if len(response.question_entities) == 0:
            logger.warning("No entities found in the question.")
        if len(response.context_entities) == 0:
            logger.warning("No entities found in the context.")
        if len(response.answer_entities) == 0:
            logger.warning("No entities found in the answer.")

        logger.info(f"Question Entities: {response.question_entities}")
        logger.info(f"Context Entities: {response.context_entities}")
        logger.info(f"Answer Entities: {response.answer_entities}")
        return response.question_entities, response.context_entities, response.answer_entities