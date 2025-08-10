# Bug Priority Classifier using BERT + Streamlit

This project predicts the priority level of software bug reports using a fine-tuned BERT model.
It features an interactive Streamlit app for EDA and live predictions. The dataset used in this project is taken from repo  [bug database](https://github.com/av9ash/gitbugs)
Each dataset has many columns **Summary, ISSUEID, Status, Resolution, Priority, Created, Resolved, Affected Versions(Only for Hadoop Database), Description**


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
**The main goal of this project is to predict the Priority of any incoming/ new data bug**. 

# Features
- Multi-class classification using BERT (DistilBERT / TFBertForSequenceClassification)
- Handles class imbalance with oversampling + focal loss
- Interactive EDA with Seaborn & Matplotlib
- Real-time prediction interface via Streamlit
- Downloadable from Google Drive for large model files

#  Tech Stack
- Python, TensorFlow, Transformers (HuggingFace)
- scikit-learn, imbalanced-learn
- Streamlit, Seaborn, Matplotlib
- Google Drive integration with gdown

#  EDA Visualizations
- Bug trends per year
- Priority distribution
- Resolved issues by priority

#  How to Run
1. Clone the repo:
git clone https://github.com/<your_username>/BugCLASSIFICATION.git
cd BugCLASSIFICATION/app
