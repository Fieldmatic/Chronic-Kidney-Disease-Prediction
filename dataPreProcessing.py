def random_value_imputation(df, feature):
    random_sample = df[feature].dropna().sample(df[feature].isna().sum())
    random_sample.index = df[df[feature].isnull()].index
    df.loc[df[feature].isnull(), feature] = random_sample


def impute_mode(df, feature):
    mode = df[feature].mode()[0]
    df[feature] = df[feature].fillna(mode)
