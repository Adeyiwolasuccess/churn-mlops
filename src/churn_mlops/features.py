import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds three engineered features: Avg_Trans_Size, Age_Group, and
    Relationship_Contact_Ratio.
    """
    df['Avg_Trans_Size'] = df['Total_Trans_Amt'] / df['Total_Trans_Ct']

    bins = [0, 35, 50, 65, 100]
    labels = ['Early Career', 'Established/Mid-Career', 'Pre-Retirement', 'Retirement Age']
    df['Age_Group'] = pd.cut(df['Customer_Age'], bins=bins, labels=labels, right=False)

    df['Relationship_Contact_Ratio'] = df['Total_Relationship_Count'] / (df['Contacts_Count_12_mon'] + 1)

    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encoding Categorical Columns with meaningful order
    """

    # Education Level

    enc = OrdinalEncoder(categories=[['Unknown', 'Uneducated', 'High School', 'College', 'Graduate', 'Post-Graduate', 'Doctorate']])
    df['Education_Level'] = enc.fit_transform(df[['Education_Level']])

    # Income Category 
    
    enc = OrdinalEncoder(categories=[['Unknown', 'Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K', '$120K +']])
    df['Income_Category'] = enc.fit_transform(df[['Income_Category']])

    # Age Group
    enc = OrdinalEncoder(categories=[['Early Career', 'Established/Mid-Career', 'Pre-Retirement', 'Retirement Age']])
    df['Age_Group'] = enc.fit_transform(df[["Age_Group"]])

    # One-Hot Encoding 
    df = pd.get_dummies(df, columns=['Gender', 'Marital_Status', 'Card_Category'], drop_first=True)

    return df


# Feature Scaling 

def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature scaling: Normalizing the range of independent variable using standardization
    """
    feature_columns = [
    'Customer_Age',
    'Dependent_count',
    'Months_on_book',
    'Total_Relationship_Count',
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Credit_Limit',
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio',
    'Avg_Trans_Size',
    'Relationship_Contact_Ratio'
    ]

    # 1. Select raw features
    features = df[feature_columns]

    # 2. Scale
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # 3. Put scaled values back into df
    scaled_df = pd.DataFrame(scaled_features, columns=feature_columns, index=features.index)
    df[feature_columns] = scaled_df

    # 4. Refresh features so it reflects the scaled values
    features = df[feature_columns]

    return df 