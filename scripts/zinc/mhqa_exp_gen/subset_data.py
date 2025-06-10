import pandas as pd

# Load the reference file
df_b = pd.read_csv('/home/sracha/proper_code_base_pubmed_dataset/datasets/mhqa-b-all-labels.csv')

# Loop through each of the five files
for i in range(1, 6):
    file_name = f'/home/sracha/proper_kg_project/scripts/mhqa_exp_gen/mhqa_{i}_with_explanations.csv'
    
    # Read the file
    df_i = pd.read_csv(file_name)
    
    # Keep only rows where the question is in df_b['question']
    df_i_filtered = df_i[df_i['question'].isin(df_b['question'])]
    
    # Save the filtered DataFrame back to the same file
    df_i_filtered.to_csv(file_name, index=False)
    print(i, ' inital: ', len(df_i), ' filtered: ', len(df_i_filtered))