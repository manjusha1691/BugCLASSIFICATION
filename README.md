# Bug Priority Classifier using BERT + Streamlit

- This project predicts the priority level of software bug reports using a fine-tuned BERT((Bidirectional Encoder Representations from Transformers) model.
- It features an interactive Streamlit app  that allows users to explore the dataset and make live predictions for the priority of the issue that has been detected
- **The main goal of this project is to predict the Priority of any incoming/ new data bug**. 


**Motivation**
In large-scale software development, efficiently managing and prioritizing bug reports is crucial. Manual classification of bugs can be time-consuming and prone to human error. It is important that each bug is classified correctly with its priority so that developers can address the most critical issues promptly. Without this prioritization, all bugs might be treated the same, potentially leading to important issues being ignored, or one bug being worked on while causing regression of another. By automating the classification process, development teams can focus on resolving high-priority bugs faster, improving software quality and user satisfaction.

**Dataset**
-  The dataset used in this project is taken from repo  [bug database](https://github.com/av9ash/gitbugs)
- Each dataset has many columns **Summary, ISSUEID, Status, Resolution, Priority, Created, Resolved, Affected Versions(Only for Hadoop Database), Description**


| Columnn | Description |
|------------------|------------------|
| Summary     | Summary of each issue    |
| Issuee ID | Identifier of the big||
| Status |The current state of the issue (e.g., Open, Resolved).
| Priority    | The assigned importance level (e.g., Blocker, Critical).     |
| Resolution    | The outcome of the issue (e.g., Fixed, Duplicate). Some values are missing.     |
| Created     | The timestamp when the issue was created.   |
| Resolved   | The timestamp when the issue was resolved.     |
| Affects Version/s|  The Hadoop version(s) impacted by the bug.|
| Description | A detailed description of the bug|
- Each dataset may have variations in the columns, with some containing additional information like "Affected Versions" for specific projects such as Hadoop.

# Features
- Multi-class classification using BERT (DistilBERT / TFBertForSequenceClassification)
- Handles class imbalance with oversampling + focal loss
- Interactive EDA with Seaborn & Matplotlib
- Real-time prediction interface via Streamlit
- Downloadable from Google Drive for large model files

 # ** Why BERT  for Bug Classification**
- BERT is used in this project because it excels at understanding the context of words in bug report descriptions, which often contain complex and technical language.
- Its bidirectional attention mechanism helps capture nuanced meanings, making it ideal for accurately classifying bug priorities.
-  By fine-tuning BERT on the bug report dataset, we can leverage its pre-trained language capabilities to improve classification accuracy.
- This enables fast, reliable priority predictions for bug reports, helping developers focus on critical issues.

# ##  Detailed Explanation of Files


### 1. `dataprep.py`
This file contains methods for loading, cleaning, and preparing bug report data. It includes functions to load CSV files from a ZIP archive, preprocess the data for analysis, and format it for model training.

#### **Key Functions:**

- **`prepareData(data)`**:
   - **Purpose**: Prepares the bug report data for analysis by:
     - Converting date columns (`Created` and `Resolved`) to `datetime`.
     - Combining the `Summary` and `Description` into a new `Synopsis` column.
     - Filling missing values in `Resolution` with "Later".
     - Dropping unnecessary columns (`Summary`, `Description`, `Affects Version/s`).
   - **Output**: Returns the cleaned and preprocessed DataFrame.

- **`load_bug_data_from_zip(zip_path)`**:
   - **Purpose**: Loads all CSV files from a ZIP archive and stores them in a dictionary.
     - Each CSV file within the ZIP file is loaded into a pandas DataFrame.
     - The dictionary keys are the folder names, and the values are the corresponding DataFrames.
   - **Output**: A dictionary containing DataFrames for each bug report dataset.

## 2 EDA: Exploratory Data Analysis for Bug Reports 'bugdata_eda.py'

This file contains various functions to perform exploratory data analysis (EDA) on the bug report dataset. The goal is to analyze key trends such as issue resolution time, issue creation by priority, and the relationship between resolved issues and priority. Below are the key functions:

### 1. **`calculateAvgIssuePeryear(data)`**
   - **Purpose**: Calculates the average resolution time of issues each year.
   - **Output**: A bar plot showing the average number of days taken to resolve bugs per year.

### 2. **`issues_per_year_priority_linechart(data)`**
   - **Purpose**: Analyzes the number of issues created each year, broken down by priority.
   - **Output**: A line chart that visualizes the trend of issues raised per year, categorized by priority.

### 3. **`resolved_issues_by_priority(data)`**
   - **Purpose**: Shows the number of issues resolved each year, grouped by priority.
   - **Output**: A bar plot illustrating the count of resolved issues per priority.

### 4. **`issues_per_year(data)`**
   - **Purpose**: Analyzes the total number of issues created each year.
   - **Output**: A bar plot representing the count of issues created annually.

### 5. **`issues_per_year_perPriority(data)`**
   - **Purpose**: Displays the number of issues created per year, categorized by their priority.
   - **Output**: A bar plot showing the number of issues created each year, divided by priority.

### 6. **`priority_vs_resolved(data)`**
   - **Purpose**: Analyzes the number of issues resolved by priority, comparing them to the total number of issues created.
   - **Output**: A bar plot displaying the number of resolved issues per priority.

### 3. 'Classification Task' `classification.py`
This file contains methods for training the BERT classifier, oversampling the data, and evaluating model performance. It includes functions for handling imbalanced datasets, training the model, and visualizing its performance.

#### **Key Functions:**

- **`oversample_text_data(X, y, random_state=42)`**: 
   - **Purpose**: Balances the dataset by oversampling underrepresented classes using the `resample` method. This helps the model correctly classify each bug priority.
   - **Output**: Returns the balanced `X` and `y` datasets.

- **`tokenize_data(X, y)`**:
   - **Purpose**: Tokenizes the bug report data for BERT. Converts text data into input format compatible with the BERT model by creating input IDs and attention masks.
   - **Output**: Returns a TensorFlow dataset that can be fed into the model.

- **`focal_loss(gamma=2.0, alpha=0.25)`**:
   - **Purpose**: Implements **Focal Loss**, which focuses on hard-to-classify examples. This loss function is particularly useful for dealing with class imbalance.
   - **Output**: A custom loss function used for model training.

- **`train_bert_classifier(X_train, y_train, X_test, y_test, num_labels, batch_size=32, epochs=2)`**:
   - **Purpose**: Fine-tunes the pre-trained BERT model on the bug report data to classify issues by priority. It uses Focal Loss to handle class imbalance and employs early stopping and learning rate reduction to optimize training.
   - **Output**: Returns the trained model and the training history.

- **`prepare_data_for_prediction(X, y, tokenizer, batch_size=32)`**:
   - **Purpose**: Prepares the dataset for making predictions. Tokenizes the data and converts it into TensorFlow format suitable for BERT.
   - **Output**: Returns a batched dataset for model prediction.

- **`predict_testdata(model, X_test, y_test, batch_size=32)`**:
   - **Purpose**: Makes predictions on the test data using the fine-tuned BERT model.
   - **Output**: Returns the predicted labels for the test dataset.

- ## 📂 Main file-  `app.py`
This file is the entry point of the Streamlit application, handling data visualization, model loading, and classification functionalities. It allows users to explore bug report data, visualize EDA plots, and classify bug descriptions into priority levels using a fine-tuned BERT model.

#### **Key Sections and Functions:**

- **Bug Data Loading & Preparation**:
   - **`load_bug_data_from_zip(zip_path)`**: Loads bug data from a ZIP file and prepares it for analysis.
   - **`prepareData(data)`**: Cleans and preprocesses the dataset by handling missing values, creating a 'Synopsis' column, and dropping unnecessary columns.

- **Model Loading**:
   - **`download_and_extract_model_from_drive()`**: Downloads and extracts the pre-trained BERT model from Google Drive.
   - **`load_finetuned_model()`**: Loads the fine-tuned BERT model and tokenizer, used for classifying bug descriptions.
   - **`save_model_and_tokenizer(model, tokenizer)`**: Saves the trained model and tokenizer for future use.

- **User Interface**:
   - **EDA Tab**: Displays various exploratory data analysis (EDA) plots such as average resolution time, issues per year by priority, and resolved issues by priority. Users can select the type of plot to visualize.
   - **Classifier Tab**: Allows users to input a bug description and classify it using the fine-tuned BERT model. If the model is not already loaded, it loads the model and predicts the bug's priority.

- **Classification Logic**:
   - **Bug Description Prediction**: Once a user inputs a bug description, the model classifies it into one of the predefined priorities (e.g., High, Medium, Low). The result is shown with the predicted priority.

This file integrates bug report analysis, model training, and interactive classification, making it easy for users to classify bug descriptions and visualize the underlying data.


## 🛠Installation & Usage

### Prerequisites
To run this project, you need the following:

- Python 3.7+
- pip (Python package installer)

### Installation Steps

1. **Clone the Repository**:
   Open your terminal or command prompt and run the following command to clone the repository to your local machine:
   ```bash
   git clone https://github.com/manjusha1691/BugCLASSIFICATION.git
   cd BugCLASSIFICATION
   pip install -r requirements.txt
   streamlit run app.py
   Access the Application:
After running the command, the application will open in your default web browser, where you can interact with the bug classification interface. 
2. ** Deployed Application **

The Bug Classification app is also deployed and accessible online.

[Bug Classification with BERT - Live Demo](https://bugclassificationwithbert.streamlit.app/)

https://bugclassificationwithbert.streamlit.app/
