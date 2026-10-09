import pandas as pd

from churn_mlops.clean import drop_leaky_columns

LEAKY_1 = 'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1'
LEAKY_2 = 'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'


def test_drop_leaky_columns_removes_only_leaky_columns():
    # 1. Build a small fake DataFrame: 2 leaky columns + 2 ordinary ones
    df = pd.DataFrame({
        'Customer_Age': [30, 45, 60],
        'Attrition_Flag': ['Existing Customer', 'Attrited Customer', 'Existing Customer'],
        LEAKY_1: [0.1, 0.2, 0.3],
        LEAKY_2: [0.9, 0.8, 0.7],
    })

    # 2. Call the function
    result = drop_leaky_columns(df)

    # 3. The leaky columns must be gone
    assert LEAKY_1 not in result.columns
    assert LEAKY_2 not in result.columns

    # 4. The ordinary columns must survive
    assert 'Customer_Age' in result.columns
    assert 'Attrition_Flag' in result.columns