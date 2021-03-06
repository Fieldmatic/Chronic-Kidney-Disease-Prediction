import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
import warnings

from dataPreProcessing import random_value_imputation, impute_mode
from decisionTreeClassifier import decision_tree_classify
from extraTreesClassifier import extra_trees_classify
from featureEncode import encode_features
from kNN import knn_classify
from randomForestClassifier import random_forest_classify

warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')
pd.set_option('display.max_columns', 26)

global X_train, X_test, Y_train, Y_test, df, cat_cols, num_cols


def load_dataset():
    global df
    df = pd.read_csv('./Dataset/kidney_disease.csv')
    df.drop('id', axis=1, inplace=True)
    df.columns = ['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 'red_blood_cells', 'pus_cell',
                  'pus_cell_clumps', 'bacteria', 'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
                  'potassium', 'haemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count',
                  'hypertension', 'diabetes_mellitus', 'coronary_artery_disease', 'appetite', 'peda_edema',
                  'aanemia', 'class']
    global cat_cols
    cat_cols = [col for col in df.columns if df[col].dtype == 'object']
    global num_cols
    num_cols = [col for col in df.columns if df[col].dtype != 'object']


def convert_to_numeric():
    df['packed_cell_volume'] = pd.to_numeric(df['packed_cell_volume'], errors='coerce')
    df['white_blood_cell_count'] = pd.to_numeric(df['white_blood_cell_count'], errors='coerce')
    df['red_blood_cell_count'] = pd.to_numeric(df['red_blood_cell_count'], errors='coerce')


def replace_incorrect_values():
    df['diabetes_mellitus'].replace(to_replace={'\tno': 'no', '\tyes': 'yes', ' yes': 'yes'}, inplace=True)
    df['coronary_artery_disease'] = df['coronary_artery_disease'].replace(to_replace='\tno', value='no')
    df['class'] = df['class'].replace(to_replace={'ckd\t': 'ckd', 'notckd': 'not ckd'})
    df['class'] = df['class'].map({'ckd': 0, 'not ckd': 1})
    df['class'] = pd.to_numeric(df['class'], errors='coerce')


def build_model():
    global X_train, X_test, Y_train, Y_test
    ind_col = [col for col in df.columns if col != 'class']
    dep_col = 'class'
    x = df[ind_col]
    y = df[dep_col]
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.30, random_state=0)


def preProcess_data():
    for col in num_cols:
        random_value_imputation(df, col)

    random_value_imputation(df, 'red_blood_cells')
    random_value_imputation(df, 'pus_cell')

    for col in cat_cols:
        impute_mode(df, col)


if __name__ == '__main__':
    load_dataset()
    convert_to_numeric()
    replace_incorrect_values()
    preProcess_data()
    encode_features(df, cat_cols)
    build_model()

    print(df.head())

    knn_acc = knn_classify(X_train, Y_train, X_test, Y_test)
    decision_tree_acc = decision_tree_classify(X_train, Y_train, X_test, Y_test)
    random_forest_acc = random_forest_classify(X_train, Y_train, X_test, Y_test)
    extra_trees_acc = extra_trees_classify(X_train, Y_train, X_test, Y_test)

    models = pd.DataFrame({'Model': ['KNN', 'Decision Tree', 'Random Forest', 'Extra Trees'],
                           'Score': [knn_acc, decision_tree_acc, random_forest_acc, extra_trees_acc]})
    models.sort_values(by='Score', ascending=False)

    bar = px.bar(data_frame=models, x='Score', y='Model', color='Score', template='plotly_dark',
                 title='Models Comparison')
    bar.show()
