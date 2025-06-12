from src.utils import OpenAIModel
from src.types.structured_response import SymptomPhraseLabel

class SymptomPhraseLabeller:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model = OpenAIModel(model_name, 'cpu')
        self.prompt = """You are a medical and mental health expert. Your goal is to assign label of between "Anxiety" or "Depression" or "Both Anxiety and Depression" to the phrase by carefully analyzing the context given.

These phrases are extracted from a reddit post of the patient.

Input Format:
Phrase: <symptom phrase which could help in identify the symptom the patient might have>
Reddit Post Label: <label which is given to the original reddit post>

Task Instruction:
- You will be given a phrase and a reddit post label.
- Also consider the reddit post label that could help you in assigning the label to the phrase.
- In the context, you will be given triplets retrieved from UMLS knowledge graph and paragraphs from some standard mental health textbooks.
- Carefully go through the context and determine the label of the phrase.
- Do NOT always assign Reddit Post Label to the phrase. It could be possible that overall reddit post label is not correct. 
- For example, if the Reddit Post Label is "Comorbid (Depression + Anxiety)" then it could be possible that some phrase are only "Anxiety" and other phrases are only "Depression", so you should assign the label accordingly.
- Another example, if the Reddit Post Label is "Depression" then it could be possible that some phrase are only "Anxiety" and other phrases are only "Depression", so you should assign the label accordingly.

Label the symptom phrases based on the patient's text.
Input:
{query}

Context:
{context}

Output:
"""
    
    def run(self, query: str, context: str) -> SymptomPhraseLabel:
        response = self.model.generate_text(self.prompt.format(query=query, context=context), structured_format=SymptomPhraseLabel, max_tokens=2048)
        return response
        