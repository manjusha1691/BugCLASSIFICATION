import os
import pandas as pd
import zipfile
import warnings
warnings.simplefilter(action='ignore')
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
# Import methods from eda file
from bugdata_eda import calculateAvgIssuePeryear, issues_per_year, issues_per_year_perPriority, resolved_issues_by_priority, priority_vs_resolved, issues_per_year_priority_linechart
#Import methods from classification file
from classification import train_bert_classifier, oversample_text_data, predict_testdata, plot_confusion_matrix, plot_training_history


""" Method to loop over the zip file and store the dataset to a dictionary"""
import pandas as pd
#Method to Convert data to a usable format

def prepareData(data):
    data['Created'] = pd.to_datetime(data['Created'])
    data['Resolved'] = pd.to_datetime(data['Resolved'])
    data['Issue id'] = data['Issue id'].astype(str)
    data['Synopsis'] = data['Summary'].fillna('').astype(str).str.strip() + ' ' + \
                       data['Description'].fillna('').astype(str).str.strip()
    data['Resolution'].fillna('Later')
    data['Resolved'].fillna(data['Resolved'].max())
    cols_to_drop = ['Summary', 'Description', 'Affects Version/s']
    existing_cols = [col for col in cols_to_drop if col in data.columns]
    data = data.drop(columns=existing_cols, axis=1)
    return data


def load_bug_data_from_zip(zip_path):
    """
    Loads all CSV files from a ZIP archive and returns a dictionary
    
    """
    allbugs_data = {}
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file in zip_ref.namelist():
            if file.startswith('bugsdata/') and file.endswith('.csv'):
                folder_name = file.split('/')[1]
                with zip_ref.open(file) as f:
                    df = pd.read_csv(f)
                    allbugs_data[folder_name] = df

  
    return allbugs_data

