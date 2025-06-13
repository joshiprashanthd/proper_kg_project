from src.utils import OpenAIModel
from src.types import Conversation

class AngstConversation:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model = OpenAIModel(model_name, 'cpu')
        self.prompt = """You are a medical and mental health expert. Your goal is to generate a real world conversation between user and assistant.

You will receive the following information in <query> tag and <context> tag:
- In <query>, a reddit post text and label identifying whether the post is about Anxiety or Depression.
- In <context>, some phrases from the reddit post itself. Each phrase will have:
    - possible symptom it conveys (short 3 to 4 words)
    - analysis about what the phrase is about
    - phrase label which tells you whether the phrase is about Anxiety or Depression
    - justification of why the phrase is about Anxiety or Depression

Conversation Characteristics:
- Symptoms Identification: User asks about symptoms and assistant should focus on gathering details about the user's current feelings, physical sensations, or observations.
- Disease/Disorder Identification: User asks about disease/disorder and assistant should attempt to connect the described symptoms to a potential condition or type of issue.
- Follow-up Questions: Assistant should ask clarifying or deeper questions to get more specific information about something already mentioned.
- Suggestions/Recommendations/Solutions: User asks about suggestions/recommendations/solutions and assistant should provide actionable advice, next steps, or potential remedies.
- Explanations: User asks about explanations and assistant should clearly describe a concept, process, or the reasoning behind a suggestion.

Task Instructions:
- Create a conversation between user and assistant.
- A turn consist of user query and assistant response.
- You should ONLY use the possible symptom, analysis, label associated to phrase and justification given to you in the <context> tags to generate the assistant response.
- Create a conversation that has subset of the characteristics mentioned above. Depending on the characteristic, carefully design the query of the user as well. 
- Do not always use the phrases given in the input, they are just for reference, you can manipulate them or just don't use them at all. The goal is to effectively use context only given for each phrase and figuring out what real people might say in a conversation.
- Create 4 turn conversation.
- User is curious about the situation and wants to know more about it.
- ASSISTANT SHOULD NOT ALWAYS ASK FOLLOW-UP QUESTIONS. Sometimes it just answer what user is asking and move to the next turn. For example, when user asks about symptoms, assistant should answer about symptoms and move to the next turn. Another example, when user asks about suggestions, assistant should answer about suggestions and move to the next turn.

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