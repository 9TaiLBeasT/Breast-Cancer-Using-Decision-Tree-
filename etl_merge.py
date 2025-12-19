
import pandas as pd
import os

def merge_datasets():
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data1_path = os.path.join(base_dir, 'Dataset', 'data_1.csv')
    data2_path = os.path.join(base_dir, 'Dataset', 'data_2.csv')
    output_path = os.path.join(base_dir, 'Dataset', 'merged_data.csv')

    print(f"Reading {data1_path}...")
    try:
        df1 = pd.read_csv(data1_path)
    except FileNotFoundError:
        print(f"Error: Could not find {data1_path}")
        return

    print(f"Reading {data2_path}...")
    try:
        df2 = pd.read_csv(data2_path)
    except FileNotFoundError:
        print(f"Error: Could not find {data2_path}")
        return

    # Transformation for df2
    # 1. Rename 'y' to 'diagnosis'
    # 2. Rename columns by removing 'x.' prefix and mapping specific suffixes
    
    rename_map = {'y': 'diagnosis'}
    for col in df2.columns:
        if col.startswith('x.'):
            new_col = col.replace('x.', '')
            # Handle specific mismatches
            if 'concave_pts' in new_col:
                new_col = new_col.replace('concave_pts', 'concave points')
            if 'fractal_dim' in new_col:
                new_col = new_col.replace('fractal_dim', 'fractal_dimension')
            rename_map[col] = new_col
            
    df2_renamed = df2.rename(columns=rename_map)
    
    # Drop "Unnamed: 0" or empty index column if it exists in df2 (column 0 is just an index)
    # Based on view_file, the first column in data_2.csv is just an index without a header name in line 1 (""), but has values "1", "2", etc.
    # Pandas read_csv often handles "","x.radius..." by naming the first col "Unnamed: 0".
    
    if "Unnamed: 0" in df2_renamed.columns:
        print("Dropping 'Unnamed: 0' from df2")
        df2_renamed = df2_renamed.drop(columns=["Unnamed: 0"])

    # Align columns: Ensure df2 has same columns as df1 (except maybe id)
    # df1 has 'id' which might not be in df2. df2 has 'diagnosis' (from y).
    
    print("Verifying column alignment...")
    # Make sure we only capture columns that exist in both (intersection) to be safe, 
    # but strictly we want to keep all valid features.
    
    # Check if 'id' is in df2, probably not. We can generate IDs or ignore.
    # The model training script drops 'id' anyway.
    
    common_cols = list(set(df1.columns).intersection(set(df2_renamed.columns)))
    
    print(f"Number of common columns: {len(common_cols)}")
    # print(f"Common columns: {common_cols}")
    
    df1_final = df1[common_cols]
    df2_final = df2_renamed[common_cols]
    
    merged_df = pd.concat([df1_final, df2_final], ignore_index=True)
    
    print(f"Merged dataset shape: {merged_df.shape}")
    
    # Save
    merged_df.to_csv(output_path, index=False)
    print(f"Merged data saved to {output_path}")

if __name__ == "__main__":
    merge_datasets()
