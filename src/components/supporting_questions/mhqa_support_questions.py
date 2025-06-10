from pydantic import BaseModel
from src.utils import OpenAIModel
from typing import List
from pydantic import Field

class QuestionOutputFormat(BaseModel):
    question_id: str = Field(description="The ID of the question")
    question: str = Field(description="The question")
    supporting_questions: List[str] = Field(description="The supporting questions")

class ResponseFormat(BaseModel):
    question_outputs: List[QuestionOutputFormat] = Field(description="The question outputs")

class MHQASupportQuestions:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model = OpenAIModel(model_name, 'cuda')
        self.prompt =  """
You are a medical expert. You are given a list of mental health related question answer pairs which are in multiple-choice format. Each QA pair has a question, four options, and a correct answer.
Your task is to generate further pairs of questions called as supporting questions based on the given QA pair.

Definitation of supporting question:
- A supporting question is a question that is related to the main question and can help in understanding the context of the main question. It should not be a direct question about the answer or options but rather a question that provides additional information or context to the main question.
- The supporting question should be descriptive/ explanatory in nature capturing various concepts and relations.
- Supporting questions can be based on the question, options, or answer depending on the context.
- Do not make a question if the context/concept/relation is already covered in the question answer.

Steps to follow:
- Read the given QA pair along with the options carefully.
- Identify concepts or keywords in the question and options that can be used to generate supporting questions.
- It is possible that the QA pair may not need any supporting question. If so do not produce any supporting question.
- Generate at max four supporting questions based on the given QA pair if needed. Do not purposely generate only four questions if the QA pair does not need any supporting question.

Examples to understand better:

<input>

question_id: 1
question: What main factor has been identified as a significant predictor for psychiatric referral in young individuals post-deliberate self-poisoning?
options: Family support, Age of the individuals, History of substance abuse, Severity of suicidal ideation
answer: Severity of suicidal ideation

question_id: 2
question: How do increased social isolation and prior mental health diagnoses relate to the risk of suicidal ideation during the COVID-19 pandemic?
options: They significantly increase the risk of suicidal ideation,They have no effect on suicidal ideation,They reduce the risk of suicidal ideation,They only affect substance use behaviors
answer: They significantly increase the risk of suicidal ideation

question_id: 3
question: What factor positively influences the outcomes of modular CBT for childhood anxiety disorders?
options: Parental involvement,Medication adherence,Age of child,Peer support
answer: Parental involvement

question_id: 4
question: What treatment combination was shown to significantly decrease anxiety sensitivity in veterans with PTSD?
options: Sertraline and placebo,Sertraline and enhanced medication management,Prolonged exposure and sertraline,Prolonged exposure and placebo
answer: Prolonged exposure and sertraline

</input>

<output>

question_id: 1
Supporting questions:
What is meant by deliberate self-poisoning in young individuals?
What is meant by psychiatric referral in the context of mental health?
What are the common factors that contribute to psychiatric referrals in young individuals?

Output:
question_id: 2
Supporting questions:
How does social isolation impact mental health during a pandemic?
What is the relationship between prior mental health diagnoses and suicidal ideation?

question_id: 3
Supporting questions:
What is meant by modular CBT in the context of childhood anxiety disorders?
How does parental involvement contribute to the effectiveness of CBT for childhood anxiety disorders?

question_id: 4
Supporting questions:
What is the significance of anxiety sensitivity in the context of PTSD?
How does prolonged exposure therapy work in treating PTSD?
What is the role of sertraline in the treatment of PTSD?
Does placebo have any effect on anxiety sensitivity for individuals with PTSD?

</output>



Read the QA pairs from the <input> tag and generate the output:

<input>

{inputs}

</input>
"""

    def generate_text(self, inputs: str) -> str:
        return self.model.generate_text(
            prompt=self.prompt.format(inputs=inputs),
            structured_format=ResponseFormat,
            max_tokens=2048
        )