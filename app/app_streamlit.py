
import streamlit as st
import pandas as pd
import numpy as np
import os

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (TFBertForSequenceClassification,
                           BertTokenizer )
from sklearn.metrics import classification_report

# Load the data
from dataprep import load_bug_data_from_zip, prepareData

#Import methods from eda file
from bugdata_eda import (calculateAvgIssuePeryear,
                              issues_per_year,
                                issues_per_year_perPriority,
                                  resolved_issues_by_priority,
                                    priority_vs_resolved,
                                      issues_per_year_priority_linechart)
#Import methods from classification file
from classification import (train_bert_classifier, 
                                oversample_text_data,
                                  predict_testdata, 
                                  plot_confusion_matrix,
                                   plot_training_history, prepare_data_for_prediction)

# Bugsdata file path
zip_path = "../data/bugsdata.zip"
# Load data from the zip file (assumes it returns a dictionary of DataFrames)
bug_data_dict = load_bug_data_from_zip(zip_path)


# Display the available keys (only once)
if "printed_keys" not in st.session_state:
    st.markdown("#### Available Bug Data Keys")
    for i, key in enumerate(bug_data_dict.keys(), 1):
        st.write(f"{i}. {key}")
    st.session_state["printed_keys"] = True

# Display dropdown to select a dataset
dataset_names = list(bug_data_dict.keys())
selected_dataset_name = st.sidebar.selectbox("Select Dataset", dataset_names)

# Prepare the selected dataset
data = prepareData(bug_data_dict[selected_dataset_name])


@st.cache_resource
def load_finetuned_model():
    model_path = "../models/bert_bug_classifier"
    st.write("Current working directory:", os.getcwd())
    if not os.path.exists(model_path):
        return None, None  # No saved model yet
    model = TFBertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    return model, tokenizer


eda_tab, classifier_tab = st.tabs([" Bug Data Analysis", "Classification"])

with eda_tab:
    st.header("Bug Data Analysis")
    eda_options = [
        "Average Resolution Time Per Year",
        "Issues Per Year by Priority",
        "Resolved Issues by Priority",
        "Issues Created Per Year",
        "Issues per Year per Priority",
        "Priority vs Resolved"
    ]
    choice = st.selectbox("Select EDA plot to view:", eda_options)

    if choice == eda_options[0]:
        fig = calculateAvgIssuePeryear(data)
        st.pyplot(fig)
    elif choice == eda_options[1]:
        fig = issues_per_year_priority_linechart(data)
        st.pyplot(fig)
    elif choice == eda_options[2]:
        fig = resolved_issues_by_priority(data)
        st.pyplot(fig)
    elif choice == eda_options[3]:
        fig = issues_per_year(data)
        st.pyplot(fig)
    elif choice == eda_options[4]:
        fig = issues_per_year_perPriority(data)
        st.pyplot(fig)
    elif choice == eda_options[5]:
        fig = priority_vs_resolved(data)
        st.pyplot(fig)

with classifier_tab:
    st.header("Bug CLassifier with  BERT Model")

    if st.button("Start Classification"):
        X = data['Synopsis']
        y = data['Priority']

        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Balance dataset
        X_balanced, y_balanced = oversample_text_data(X, y_encoded)

        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced, y_balanced, test_size=0.2, stratify=y_balanced, random_state=42
        )

        model, tokenizer = load_finetuned_model()

        if model is None or tokenizer is None:
            st.info("No saved model found. Training model now... (CPU training can take time)")
            model, _ = train_bert_classifier(X_train, y_train, X_test, y_test, num_labels=len(le.classes_))
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            st.success("Training complete and model saved.")
        else:
            st.success("Loaded fine-tuned model from disk.")

        # Prepare test data for prediction
        test_dataset = prepare_data_for_prediction(X_test, y_test, tokenizer)

        # Predict
        probs = model.predict(test_dataset)
        y_pred = np.argmax(probs.logits, axis=1)

        # Show classification report
        report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
        st.write("### Classification Report")
        st.dataframe(pd.DataFrame(report).transpose())

        # Confusion Matrix plot
        st.write("### Confusion Matrix")
        plot_confusion_matrix(y_test, y_pred, class_labels=le.classes_)





