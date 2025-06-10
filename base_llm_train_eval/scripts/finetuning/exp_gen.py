import pandas as pd
from src.utils import OpenAIModel
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import argparse

# take n as input from command line
parser = argparse.ArgumentParser(description="process split number")
parser.add_argument(
    "--n",
    type=int,
    help="The index of the split csv file to process (1-5)",
    required=True,
)

args = parser.parse_args()
n = args.n
if n < 1 or n > 5:
    raise ValueError("n must be between 1 and 5")

model = OpenAIModel("gpt-4o-mini", 'cuda')

df = pd.read_csv(f"/home/sracha/proper_kg_project/scripts/mhqa_exp_gen/split_csv/mhqa_split_{n}.csv")
sample_df = df
#sample_df = df.sample(100, random_state=42)

prompt = """You are a medical expert. You are given a question, it's four options, and also the correct answer. 
- Generate cogent and coherent explanations for the following multiple-choice questions 
- In the explanation, justify the correct answer clearly using domain knowledge.
- Briefly explain why the other options are incorrect. Use both option letters and option names in explanations if options are compared.
Explanations should be limited to no more than 120 words.

Input Format:
ID: <unique identifier of the question>
Question: <question text>
Option1: <option 1 text>
Option2: <option 2 text>
Option3: <option 3 text>
Option4: <option 4 text>
Correct Option Number: <correct option number>

Output Format:
ID: <unique identifier of the question>
Question: <question text>
Explanation: <explanation text>

Following are the question for which you have to generate explanations:
{questions}

Generate explanation for each question and answer pair.
"""

question_template = """ID: {id}
Question: {question}
Option1: {option1}
Option2: {option2}
Option3: {option3}
Option4: {option4}
Correct Option Number: {correct_option_number}"""

class QuestionResponseFormat(BaseModel):
    ID: str = Field(description="Unique identifier of the question")
    Question: str = Field(description="Question text")
    Explanation: str = Field(description="Explanation text")

class OutputResponseFormat(BaseModel):
    questions: list[QuestionResponseFormat] = Field(
        description="List of questions and their corresponding explanations"
    )
    

def generate_explanation(batch: pd.DataFrame):
    question_text = []
    for i, row in batch.iterrows():
        question_text.append(question_template.format(
            id=row['id'],
            question=row['question'],
            option1=row['option1'],
            option2=row['option2'],
            option3=row['option3'],
            option4=row['option4'],
            correct_option_number=int(row['correct_option_number'])
        ))
    
    question_text = "\n\n".join(question_text)
    newprompt = prompt.format(questions=question_text)
    result = model.generate_text(newprompt, structured_format=OutputResponseFormat)
    
    #print("RESULT: ", result)

    for i, response in enumerate(result.questions):
        sample_df.at[batch.index[i], 'explanation'] = response.Explanation
        
batch_size = 2

with ThreadPoolExecutor(max_workers=16) as executor:
    futures = []
    for i in range(0, len(sample_df), batch_size):
        batch = sample_df[i:i + batch_size]
        futures.append(executor.submit(generate_explanation, batch))
    
    for future in tqdm(futures):
        future.result()

sample_df.to_csv(f"./mhqa_{n}_with_explanations.csv", index=False)