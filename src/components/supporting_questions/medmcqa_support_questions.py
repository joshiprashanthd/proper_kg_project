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

class MedMCQASupportQuestions:
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
- Generate at max four supporting questions based on the given QA pair if needed. Do not purposely generate only four questions if the QA pair does not need any supporting question. You can generate less than four questions if the QA pair does not need any supporting question.

Examples to understand better:

<input>

question_id: 1
question: Treatment of Choice in Sleep Apnea Syndrome:
options: Sedatives.,Antidepressants.,Antiepileptics,Continuous positive pressure ventilation
answer: Continuous positive pressure ventilation

question_id: 2
question: A victim of Tsunami has difficulty in overcoming the experience. She still recollects the happening in dreams and thoughts. The most likely diagnosis is
options: Conversion disorder,Post traumatic stress disorder,Phobia,Panic disorder
answer: Post traumatic stress disorder

question_id: 3
question: Features of alcohol withdrawl are all EXCEPT:
options: Restlessness,Hypersomnolence,Hallucination,Epileptic seizure
answer: Hypersomnolence

question_id: 4
question: NREM sleep is associated with:-
options: Basal forebrain area,Medulla,Dorsal raphe nucleus,All of the above
answer: All of the above

</input>


<output>

question_id: 1
Supporting questions:
What is the pathophysiology of sleep apnea syndrome?

question_id: 2
Supporting questions:
What are the common symptoms of post traumatic stress disorder?
What are the common symptoms of conversion disorder?
What are the common symptoms of phobia?
What are the common symptoms of panic disorder?

question_id: 3
Supporting questions:
What are the common symptoms of alcohol withdrawal?

question_id: 4
Supporting questions:
What is meant by NREM sleep?
What is the role of the basal forebrain area in sleep regulation?
How does the medulla contribute to the sleep cycle?
What is the function of the dorsal raphe nucleus in relation to sleep?

</output>


Read the QA pairs from the <input> tag and generate the output:

<input>

{inputs}

</input>
"""

    def run(self, inputs: str) -> str:
        return self.model.generate_text(
            prompt=self.prompt.format(inputs=inputs),
            structured_format=ResponseFormat,
            max_tokens=2048
        )