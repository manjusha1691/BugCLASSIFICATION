
import streamlit as st
import pandas as pd
import numpy as np
import os
import gdown
import zipfile
from pathlib import Path
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import TFBertForSequenceClassification, BertTokenizer
from sklearn.metrics import classification_report

# Load utility functions for data processing and visualization
#Load data
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
##------------Load the bug data set
zip_path = Path("data/bugsdata.zip")
if not zip_path.exists():
    st.error("Zip file not found!")
else:
    bug_data_dict = load_bug_data_from_zip(zip_path)

# Display dropdown to select a dataset
dataset_names = list(bug_data_dict.keys())
selected_dataset_name = st.sidebar.selectbox("Select Dataset", dataset_names)

# Prepare the selected dataset
data = prepareData(bug_data_dict[selected_dataset_name])

#Method to download and Load Model
@st.cache_resource
def download_and_extract_model_from_drive():
    ##Downloads a zipped BERT model from Google Drive and extracts it to the 'models' directory.
    

    file_id = '1hyUdeyxhwP7zibqd1Klz18Vrbw34JxpW'  # Your Google Drive File ID
    zip_output_path = Path('models/bert_bug_classifier.zip')
    extract_dir = Path('models/bert_bug_classifier')

    # Download and extract only if the model directory doesn't already exist
    if not extract_dir.exists():
        zip_output_path.parent.mkdir(parents=True, exist_ok=True)

        # Download zip if not already present
        if not zip_output_path.exists():
            gdown.download(f'https://drive.google.com/uc?id={file_id}', str(zip_output_path), quiet=False)

        # Extract zip
        with zipfile.ZipFile(str(zip_output_path), 'r') as zip_ref:
            zip_ref.extractall(str(extract_dir))
            # After extraction, check if there is a nested folder
        extracted_contents = list(extract_dir.iterdir())
        st.write("Extracted folder contents:", [p.name for p in extracted_contents])
        if len(extracted_contents) == 1 and extracted_contents[0].is_dir():
            # If yes, update extract_dir to this nested folder
            extract_dir = extracted_contents[0]


    return extract_dir

#Loads the fine-tuned BERT model and tokenizer from the extracted model directory.
@st.cache_resource
def load_finetuned_model():
    model_dir = download_and_extract_model_from_drive()
    # Debug: List files in model directory after extraction
    import os
    st.write("Model files after extraction:", os.listdir(str(model_dir)))
    if not model_dir.exists():
        return None, None
    model = TFBertForSequenceClassification.from_pretrained(str(model_dir))
    tokenizer = BertTokenizer.from_pretrained(str(model_dir))
    return model, tokenizer


#Method to save model and tokeniser
def save_model_and_tokenizer(model, tokenizer, save_path="models/bert_bug_classifier"):
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))

#Tabls to be dispalyed in StreamlitApp
eda_tab, classifier_tab = st.tabs([" Bug Data Analysis", "Classification"])
### ------------------------EDA TAB -----------------------------------------
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
# ------- CLASSIFICATION TAB
with classifier_tab:

    st.header("Bug CLassifier with  BERT Model")
    
    bug_description = st.text_area("Enter bug description:")

    if st.button("Classify"):
       
       #If user does not entrr anything and click on Classify, then Warn the user to entr some text
       if bug_description.strip() == "":
            st.warning("Please enter a bug description.")
       else:
        #Prepare Training data for label encoder
        X = data['Synopsis']
        y = data['Priority']

        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

         # Load the fine-tuned model and tokenizer with progress feedback
        with st.spinner("Loading BERT model..."):
            model, tokenizer = load_finetuned_model()
    
        # If Model is not loaded or present, then do the training here and Classify here

        if model is None or tokenizer is None:
            st.error("❌ Model could not be loaded from Google Drive. Please ensure the model zip is correct.")
            st.stop()
        
        # Predict on input bug description
        inputs = tokenizer(bug_description, return_tensors="tf", truncation=True, padding=True, max_length=128)
        #Predict with the model
        outputs = model(inputs)
        probs = tf.nn.softmax(outputs.logits, axis=1).numpy().flatten()
        pred_index = np.argmax(probs)
        pred_label = le.inverse_transform([pred_index])[0]

        st.success(f"Predicted Priority: **{pred_label}** ({probs[pred_index]:.2%} confidence)")

        # Show all class probabilities
        #labels = le.inverse_transform(np.arange(len(probs)))
        #conf_df = pd.DataFrame({
           # "Priority": labels,
            #"Confidence": probs
        #}).sort_values("Confidence", ascending=False)

        #st.dataframe(conf_df.style.format({"Confidence": "{:.2%}"}))

           



