# Bug Priority Classifier using BERT + Streamlit

- This project predicts the priority level of software bug reports using a fine-tuned BERT((Bidirectional Encoder Representations from Transformers) model.
- It features an interactive Streamlit app  that allows users to explore the dataset and make live predictions for the priority of the issue that has been detected
- **The main goal of this project is to build a model to  predict the Priority of any incoming/ new data bug**. 


**Motivation**
- In large-scale software development, efficiently managing and prioritizing bug reports is crucial. Manual classification of bugs can be time-consuming and prone to human error. It is important that each bug is classified correctly with its priority so that developers can address the most critical issues promptly. Without this prioritization, all bugs might be treated the same, potentially leading to important issues being ignored, or one bug being worked on while causing regression of another. By automating the classification process, development teams can focus on resolving high-priority bugs faster, improving software quality and user satisfaction.

**Dataset**
- The dataset used in this project is taken from repo  [bug database](https://github.com/av9ash/gitbugs)
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

 ###  Why BERT  for Bug Classification
- BERT is used in this project because it excels at understanding the context of words in bug report descriptions, which often contain complex and technical language.
- Its bidirectional attention mechanism helps capture nuanced meanings, making it ideal for accurately classifying bug priorities.
-  By fine-tuning BERT on the bug report dataset, we can leverage its pre-trained language capabilities to improve classification accuracy.
- This enables fast, reliable priority predictions for bug reports, helping developers focus on critical issues.


## Installation & Usage

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


