
import pandas as pd

try:
    df1 = pd.read_csv('Dataset/data_1.csv')
    print("Columns in data_1.csv:")
    print(list(df1.columns))
except Exception as e:
    print(f"Error reading data_1.csv: {e}")

try:
    df2 = pd.read_csv('Dataset/data_2.csv')
    print("\nColumns in data_2.csv:")
    print(list(df2.columns))
except Exception as e:
    print(f"Error reading data_2.csv: {e}")
