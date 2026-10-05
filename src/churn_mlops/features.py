import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

FEATURE_COLUMNS = [
    'Customer_Age', 'Dependent_count', 'Months_on_book',
    'Total_Relationship_Count', 'Months_Inactive_12_mon',
    'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
    'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
    'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
    'Avg_Trans_Size', 'Relationship_Contact_Ratio'
]

ONE_HOT_COLUMNS = [
    'Gender_M', 'Marital_Status_Married', 'Marital_Status_Single',
    'Marital_Status_Unknown', 'Card_Category_Gold',
    'Card_Category_Platinum', 'Card_Category_Silver'
]

EDUCATION_ORDER = ['Unknown', 'Uneducated', 'High School', 'College',
                    'Graduate', 'Post-Graduate', 'Doctorate']
INCOME_ORDER = ['Unknown', 'Less than $40K', '$40K - $60K',
                 '$60K - $80K', '$80K - $120K', '$120K +']
AGE_GROUP_ORDER = ['Early Career', 'Established/Mid-Career',
                     'Pre-Retirement', 'Retirement Age']


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds three engineered features: Avg_Trans_Size, Age_Group, and
    Relationship_Contact_Ratio.
    """
    df['Avg_Trans_Size'] = df['Total_Trans_Amt'] / df['Total_Trans_Ct']

    bins = [0, 35, 50, 65, 100]
    df['Age_Group'] = pd.cut(df['Customer_Age'], bins=bins,
                              labels=AGE_GROUP_ORDER, right=False)

    df['Relationship_Contact_Ratio'] = (
        df['Total_Relationship_Count'] / (df['Contacts_Count_12_mon'] + 1)
    )
    return df


def fit_encoders(df: pd.DataFrame) -> dict:
    """
    Fits one OrdinalEncoder per ordinal column and returns them in a
    dict, so they can be saved and reused at prediction time.
    """
    encoders = {
        'Education_Level': OrdinalEncoder(categories=[EDUCATION_ORDER]),
        'Income_Category': OrdinalEncoder(categories=[INCOME_ORDER]),
        'Age_Group': OrdinalEncoder(categories=[AGE_GROUP_ORDER]),
    }
    for column, encoder in encoders.items():
        encoder.fit(df[[column]])
    return encoders


def apply_encoders(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """
    Applies already-fitted OrdinalEncoders to the ordinal columns, then
    one-hot encodes Gender, Marital_Status, and Card_Category, ensuring
    all expected dummy columns exist even if a category is missing
    from this particular batch of data.
    """
    for column, encoder in encoders.items():
        df[column] = encoder.transform(df[[column]])

    df = pd.get_dummies(df, columns=['Gender', 'Marital_Status', 'Card_Category'],
                         drop_first=True)

    for col in ONE_HOT_COLUMNS:
        if col not in df.columns:
            df[col] = False

    return df


def fit_scaler(df: pd.DataFrame) -> StandardScaler:
    """
    Fits a StandardScaler on the continuous feature columns.
    """
    scaler = StandardScaler()
    scaler.fit(df[FEATURE_COLUMNS])
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """
    Applies an already-fitted StandardScaler to the continuous feature
    columns, replacing them in place with their scaled values.
    """
    df[FEATURE_COLUMNS] = scaler.transform(df[FEATURE_COLUMNS])
    return df