from src.utils import OpenAIModel

class TripletRAGExplainer:
    def __init__(self):
        self.prompt_template = """You are a medical expert. Your goal is to explain the reasoning for a given medical question's answer.

You will receive the following information in <query> tag and <context> tag:
- The MCQ type medical question
- The options for the question
- The correct answer to the question
- Contextual paragraphs related to the question, options and answer.
- A set of triplets (subject, predicate, object) representing medical facts.
    - Structure of triplet: (head: '<head node name>', Definition: '<definition of head node>')-[<relation name>]->(tail: '<tail node name>', Definition: '<definition of tail node>'

Task Instructions:
- Carefully analyze every piece of information from the triplets and context.
- Look at each triplet and find out how each relation makes sense to the question and answer given.
- If a triplet is not directly related to the question and answer, clearly state that.
- Go through each paragraph given as the additional context.
- A paragraph contains information related to the question and answer that might help you find missing pieces that could lead you to the correct answer.
- The explanation should be very clear in how it connects the question to the answer.
- In the explanation, try to compare each and every option with the context and explain why the option is incorrect and why the correct answer is correct.
- Only use the information given to you in the <context> tag to generate the explanation.

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