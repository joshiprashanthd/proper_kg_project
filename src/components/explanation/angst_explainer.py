from src.utils import OpenAIModel

class AngstExplainer:
    def __init__(self):
        self.prompt_template = """You are a medical expert. Your goal is to explain how a Reddit post indicates if the user has anxiety or depression.

You will receive the following information in <query> tag and <context> tag:

- A label that tells you if the user actually has Anxiety or Depression.
- A list of sentences from a Reddit post.
- For each sentence, you will get extra information:
    - Related paragraphs: Parts of other documents that are similar to the sentence.
    - Knowledge Graph details: Short facts (like "person - has symptom - feeling tired") linked to the sentence.

Task Instructions:
- For each sentence, explain how it connects to the given label (Anxiety or Depression). Explanation should be detailed and thorough.
- Point out which sentences most strongly suggest symptoms of Anxiety or Depression.
- Address sentences that might seem unclear or could fit both conditions. Explain why they lean more towards one over the other, or if they are general symptoms.
- If a sentence does not seem related to either Anxiety or Depression, clearly state that.
- Make sure your explanation is easy to understand for someone without medical knowledge. Avoid complex medical terms.

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
