from sklearn.preprocessing import LabelEncoder


def encode_features(df, cat_cols):
    le = LabelEncoder()

    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
