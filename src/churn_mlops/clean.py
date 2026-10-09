"""
clean.py 

Handles the cleaning of the two Naive_Bayes_Classifier columns to
avoid leakeage
"""

import pandas as pd 

def drop_leaky_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
    ])
    return df