from src.utils import OpenAIModel
from src.types import Conversation

class AngstSymptomDisorderId:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model = OpenAIModel(model_name, 'cpu')
        self.prompt = """You are a medical and mental health expert. Your goal is to generate a real world conversation between user and a model.

You will receive the following information in <query> tag and <context> tag:
- In <query>, a reddit post text and label identifying whether the post is about Anxiety or Depression.
- In <context>, some phrases from the reddit post itself. Each phrase will have:
    - possible symptom it conveys (short 3 to 4 words)
    - analysis about what the phrase is about
    - phrase label which tells you whether the phrase is about Anxiety or Depression
    - justification of why the phrase is about Anxiety or Depression
    - context which retrieved from standard mental health textbooks and triplets from UMLS knowledge graph.

Task Instructions:
- Create a conversation between user and model.
- User is asking about what symptoms does he or she have by given some phrases from the reddit post.
- Model should use the possible symptom, analysis, phrase label, justification and context associated with that phrase.
- Create a 4 turn conversation.

<query>
{query}
</query>

<context>
{context}
</context>
"""

    def run(self, query: str, context: str) -> Conversation:
        response = self.model.generate_text(self.prompt.format(query=query, context=context), structured_format=Conversation)
        return response