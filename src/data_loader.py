import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(path):
    df = pd.read_csv(path)

    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["intent"])

    return df, label_encoder
