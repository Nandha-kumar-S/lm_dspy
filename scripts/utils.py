import dspy
import pandas as pd

#to get history
def get_history(lm, n):
    history = lm.history
    last_history = {}
    if len(history) >= n:
        last_history['system'] = history[-n:][0]['messages'][0]['content']
        last_history['user'] = history[-n:][0]['messages'][1]['content']
    return last_history

#process dataframe
def process_df(df, input_columns, output_column):
    # Add other steps if needed
    processed_df = df.map(lambda x: str(x).lower() if x is not None else x)
    return processed_df