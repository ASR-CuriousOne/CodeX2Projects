import numpy as np
import pandas as pd

df_raw = pd.read_csv("training_data.csv")

df_processed = {}

def ConvertToKeysValue(x):
    keyValuePairs = {}
    Length = len(x)
    for n,i in enumerate(x):
        keyValuePairs[i] = (n - (Length-1)/2)
    return keyValuePairs  

for column in df_raw:
    if column == "ID" or column == "Predicted Hobby":         
        continue    

    Keys = df_raw[column].unique()
    
    KeyValuePairs = ConvertToKeysValue(Keys)
    df_processed[column] = ([KeyValuePairs[i] for i in df_raw[column]])

df_processed["Predicted Hobby"] = df_raw["Predicted Hobby"]

df_processed = pd.DataFrame(df_processed)

df_processed.to_csv("processed_training_data.csv",index=False)