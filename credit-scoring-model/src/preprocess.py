import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(path):
    data = pd.read_csv(path)

    # Handle missing values
    data = data.dropna()

    # Encode categorical columns
    le = LabelEncoder()
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = le.fit_transform(data[col])

    return data