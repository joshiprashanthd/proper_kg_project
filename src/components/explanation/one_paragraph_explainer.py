from src.utils import OpenAIModel

class OneParagraphExplainer:
    def __init__(self):
        self.prompt_template = """You are a medical expert. Your goal is to explain the reasoning for a given medical question's answer.

You will receive the following information in <query> tag and <context> tag:
- The MCQ type medical question
- The options for the question
- The correct answer to the question
- Contextual paragraphs relating to the question, options and answer.
- A set of triplets (subject, predicate, object) representing medical facts.
    - Structure of triplet: (head: '<head node name>', Definition: '<definition of head node>')-[<relation name>]->(tail: '<tail node name>', Definition: '<definition of tail node>'

Task Instructions:
- Carefully analyze every piece of information from the triplets and context.
- Look at each triplet and find out how each relation makes sense to the question and answer given.
- If a triplet is not directly related to the question and answer, clearly state that.
- Go through each paragraph given as the additional context.
- A paragraph contains information relating to the question and answer might help you find missing pieces that could lead you to the answer starting from the question.
- The explanation should be very clear in how it connects the question to the answer.
- Only use the information given to you in the <context> tag to generate the explanation.

IMPORTANT: You explanations should written in a single paragraph. Do not break the explanation into multiple paragraphs. The explanations should be 200 to 250 words long.

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
        return self.model.generate_text(prompt, max_tokens=2048)