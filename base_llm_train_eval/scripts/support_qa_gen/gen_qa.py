from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys
sys.path.append(".")

import datetime
import pandas as pd
from models import OpenAIModel
from tqdm import tqdm
from pydantic import BaseModel

prompt_mhqa= """
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


prompt_medmcqa= """
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

def run(df: pd.DataFrame):
    batch_size = 10
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for i in range(0, len(df), batch_size):
            batch = df[i:i+batch_size]
            futures.append(executor.submit(process_batch, batch))

        for future in tqdm(as_completed(futures)):
            future.result()

if __name__ == "__main__":
    df_mhqa = pd.read_csv('/home/sracha/proper_kg_project/base_llm_train_eval/dataset/train/mhqa_b_train.csv', index_col='question_id').sample(50, random_state=42)
    df_medmcqa = pd.read_csv('/home/sracha/proper_kg_project/base_llm_train_eval/dataset/train/medmcqa_train.csv', index_col='question_id').sample(50, random_state=42)

    # df_mhqa['question_id'] = df_mhqa.index
    # df_medmcqa['question_id'] = df_medmcqa.index





