import pandas as pd

# Load the original dataset
df = pd.read_csv('/home/sracha/proper_kg_project/data/mhqa/mhqa-b_final.csv')

# Shuffle the data for random selection
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split the data into 5 roughly equal parts
split_dfs = []
split_size = len(df) // 5

for i in range(4):
    part = df_shuffled.iloc[i*split_size : (i+1)*split_size]
    split_dfs.append(part)

# Last part gets the remaining rows to ensure all rows are included
split_dfs.append(df_shuffled.iloc[4*split_size :])

# Save each part to a new CSV
for i, part in enumerate(split_dfs):
    part.to_csv(f'/home/sracha/proper_kg_project/scripts/mhqa_exp_gen/split_csv/mhqa_split_{i+1}.csv', index=False)

