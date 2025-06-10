import pandas as pd

# Load your CSV dataset
csv_path = "/home/sracha/proper_kg_project/scripts/mhqa_exp_gen/mhqa_1_with_explanations.csv"  # Replace with your actual file path
df = pd.read_csv(csv_path)
#print(df['correct_answer'][10])
# Ensure column names are correct and create the new DataFrame with proper line breaks
df1 = pd.DataFrame({
    "query": df.apply(lambda row: f"""question: {row['question']}\nOptions:\noption1: {row['option1']}\noption2: {row['option2']}\noption3: {row['option3']}\noption4: {row['option4']}""", axis=1),
    "answers": df.apply(lambda row: f"""explanation: {row['explanation']}\nanswer: {row['correct_answer']}""", axis=1)
})

print(df1['query'][1])