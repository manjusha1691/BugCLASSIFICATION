import os
import pandas as pd
import zipfile
import warnings
warnings.simplefilter(action='ignore')
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from app.clean_data import prepareData

from sklearn.model_selection import train_test_split
# Import methods from eda file
from app.bugdata_eda import calculateAvgIssuePeryear, issues_per_year, issues_per_year_perPriority, resolved_issues_by_priority, priority_vs_resolved, issues_per_year_priority_linechart
#Import methods from classification file
from app.classification import train_bert_classifier, oversample_text_data, predict_testdata, plot_confusion_matrix, plot_training_history


""" Method to loop over the zip file and store the dataset to a dictionary"""
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

    # Display available keys
    keys = list(allbugs_data.keys())
    print("Available bug data keys:")
    for i, key in enumerate(keys, 1):
        print(f"{i}. {key}")
    
    return allbugs_data

# Bugsdata file path
zip_path = "data/bugsdata.zip"
#Load the bug data files
allbugs_data = load_bug_data_from_zip(zip_path)
bugs_dataset = list(allbugs_data.keys())

# Ask user to enter the number corresponding to the dataset you want to explore
while True:
    try:
        choice = int(input(f"Enter the number (1-{len(bugs_dataset)}) to select bug data for EDA: ").strip())
        if 1 <= choice <= len(bugs_dataset):
            selected_key = bugs_dataset[choice - 1]
            break
        else:
            print(f"Please enter a number between 1 and {len(bugs_dataset)}.")
    except ValueError:
        print("Invalid input. Please enter a valid number.")

# Now load and prepare the selected data
data = allbugs_data[selected_key]
data = prepareData(data)
print(f"Selected dataset: {selected_key}")

print("\nWhat do you want to perform?")
print("1. Exploratory Data Analysis (EDA)")
print("2. Classification (BERT)")

task_choice = int(input("Enter 1 for EDA or 2 for Classification: "))

if task_choice == 1:

    calculateAvgIssuePeryear(data)
    issues_per_year_priority_linechart(data)
    resolved_issues_by_priority(data)
    issues_per_year(data)
    issues_per_year_perPriority(data)
    priority_vs_resolved(data)
elif task_choice == 2:
    print(f"Running Classification on {selected_key} ...")
    print("OverSAmpling the DAta")
    X = data['Synopsis']
    y = data['Priority']

    # Encode target labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X_balanced, y_balanced = oversample_text_data(X, y_encoded)

    # Split into 80% train and 20% test
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, stratify=y_balanced, random_state=42)

    print("Done OverSAmpling and Splitting")
    # Step 2: Train model without val split
    print("Training BERT model...")
    model, finetuned_model= train_bert_classifier(X_train, y_train, X_test, y_test, num_labels=len(le.classes_))
    print("Model training complete.")

    # Step 3: Predict
    y_pred = predict_testdata(model, X_test, y_test, batch_size=32)
    print("Prediction complete.")

    #      Step 4: Confusion matrix
    plot_confusion_matrix(y_test, y_pred, le.classes_)
    print("Confusion matrix plotted.")

    # Step 5: Training history
    plot_training_history(finetuned_model)
    print("Training history plotted.")

