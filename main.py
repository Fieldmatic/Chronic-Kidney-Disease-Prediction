import pandas as pd

def loadDataSet():
    global df, df
    df = pd.read_csv('./Dataset/kidney_disease.csv')
    df.head()
    df.drop('id', axis=1, inplace=True)
    df.columns = ['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 'red_blood_cells', 'pus_cell',
                  'pus_cell_clumps', 'bacteria', 'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
                  'potassium', 'haemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count',
                  'hypertension', 'diabetes_mellitus', 'coronary_artery_disease', 'appetite', 'peda_edema',
                  'aanemia', 'class']


def convert_to_numeric():
    df['packed_cell_volume'] = pd.to_numeric(df['packed_cell_volume'], errors='coerce')
    df['white_blood_cell_count'] = pd.to_numeric(df['white_blood_cell_count'], errors='coerce')
    df['red_blood_cell_count'] = pd.to_numeric(df['red_blood_cell_count'], errors='coerce')


def replace_incorrect_values():
    global cat_cols, num_cols
    cat_cols = [col for col in df.columns if df[col].dtype == 'object']
    num_cols = [col for col in df.columns if df[col].dtype != 'object']
    df['diabetes_mellitus'].replace(to_replace={'\tno': 'no', '\tyes': 'yes', ' yes': 'yes'}, inplace=True)
    df['coronary_artery_disease'] = df['coronary_artery_disease'].replace(to_replace='\tno', value='no')
    df['class'] = df['class'].replace(to_replace={'ckd\t': 'ckd', 'notckd': 'not ckd'})
    df['class'] = df['class'].map({'ckd': 0, 'not ckd': 1})
    df['class'] = pd.to_numeric(df['class'], errors='coerce')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    loadDataSet()
    replace_incorrect_values()
    convert_to_numeric()
