import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np

'''
df1=pd.read_csv('/home/sracha/proper_kg_project/data/qna/combined/test/mhqa_test.csv')
df2=pd.read_csv('/home/sracha/proper_kg_project/scripts/qa_evaluation/results/reasoning_llm/mhqa_llama_results.csv')

# add 'answer' column from df1 to df2. it should be merged on 'id' column
df2 = df2.merge(df1[['question', 'correct_option_number']], on='question', how='left')
# rename the 'answer' column to 'ground_truth'
df2.rename(columns={'answer': 'correct_option_number'}, inplace=True)
#save the df2 to by replacing the original file
df2.to_csv('/home/sracha/proper_kg_project/scripts/qa_evaluation/results/reasoning_llm/mhqa_llama_results.csv', index=False)
'''

df = pd.read_csv('/home/sracha/proper_kg_project/scripts/qa_evaluation/results/reasoning_llm/medmcqa_llama_results.csv')
# script to find f1, precision, recall in df between df['gold'] and df['answer']

def f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')
def precision(y_true, y_pred):
    return precision_score(y_true, y_pred, average='macro')
def recall(y_true, y_pred):
    return recall_score(y_true, y_pred, average='macro')

# ...existing code...

# Create mapping for numerical values to letters
def map_to_letter(x):
    if x == 0 or x == -1:
        return 'A'
    elif x == 1:
        return 'B'
    elif x == 2:
        return 'C'
    elif x == 3:
        return 'D'
    return x

# Convert gold labels to letters
df['gold'] = df['gold'].apply(map_to_letter)

# Filter valid answers
valid_answers = ['A', 'B', 'C', 'D']
df_filtered = df[df['answer'].isin(valid_answers)]

# Calculate scores
y_true = df_filtered['gold']
y_pred = df_filtered['answer']

f1_score = f1(y_true, y_pred)
precision_score = precision(y_true, y_pred)
recall_score = recall(y_true, y_pred)

print(f"F1 Score: {f1_score:.3f}")
print(f"Precision: {precision_score:.3f}")
print(f"Recall: {recall_score:.3f}")
print(f"Total valid samples evaluated: {len(df_filtered)} out of {len(df)} total samples")

