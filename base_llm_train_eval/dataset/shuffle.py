import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

def shuffle(df):
    shuffled_df = df.copy()
    
    for idx in tqdm(range(len(df))):
        options = [df.loc[idx, f'option{i}'] for i in range(1, 5)]
        correct_num = int(df.loc[idx, 'correct_option_number'].item())
        correct_answer = options[int(correct_num - 1)]
        
        np.random.shuffle(options)
        
        for i in range(4):
            shuffled_df.loc[idx, f'option{i+1}'] = options[i]
        
        # Find the new position of the correct answer
        new_correct_num = options.index(correct_answer) + 1
        shuffled_df.loc[idx, 'correct_option_number'] = new_correct_num
    
    return shuffled_df

if __name__ == "__main__":
    np.random.seed(42)

    parser = argparse.ArgumentParser(description="Shuffle options in a CSV file.")
    parser.add_argument("input_file", type=str, help="Path to the input CSV file")
    parser.add_argument("--output_file", type=str, help="Path to the output CSV file",  required=False)
    
    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = args.input_file.replace(".csv", "_shuffled.csv")
    
    df = pd.read_csv(args.input_file)
    shuffled_df = shuffle(df)

    shuffled_df.to_csv(args.output_file, index=False)
    
    print("Distribution of correct options after shuffling:")
    print(shuffled_df['correct_option_number'].value_counts().sort_index())