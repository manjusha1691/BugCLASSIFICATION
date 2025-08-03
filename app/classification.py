#Import NEcessary libraries
# Libraries to help with reading and manipulating data
import numpy as np
import pandas as pd
# Libraries to help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns

#Libraries for anlyse teh datsa
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample


import os

import shutil
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
#Libraries for Model Generation
from transformers import logging
logging.set_verbosity_error()

from transformers import BertTokenizer, TFBertModel, TFBertForSequenceClassification
from transformers import BertConfig
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


"""Method to over sample the data to balance the labels of teh data set and classify the data Correctly"""

def oversample_text_data(X, y, random_state=42):
    X = np.array(X)
    y = np.array(y)
    unique_classes, class_counts = np.unique(y, return_counts=True)
    majority_count = class_counts.max()

    X_oversampled, y_oversampled = [], []

    for cls in unique_classes:
        X_cls = X[y == cls]
        y_cls = y[y == cls]
        if len(X_cls) < majority_count:
            X_up, y_up = resample(X_cls, y_cls, replace=True, n_samples=majority_count, random_state=random_state)
        else:
            X_up, y_up = X_cls, y_cls

        X_oversampled.append(X_up)
        y_oversampled.append(y_up)

    X_balanced = np.concatenate(X_oversampled)
    y_balanced = np.concatenate(y_oversampled)

    # Shuffle
    idx = np.random.permutation(len(X_balanced))
    return X_balanced[idx], y_balanced[idx]

"""**Split Train and Test Data**"""

"""Method to Tokenise the data for BERT"""

# ========== Tokenization and Dataset Preparation ==========
def tokenize_data(X, y):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encodings = tokenizer(list(X), truncation=True, padding=True, max_length=128)
    dataset = tf.data.Dataset.from_tensor_slices((dict(encodings), y))
    return dataset

# ========== Focal Loss ==========
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.reshape(tf.cast(y_true, tf.int32), [-1])
        y_true_onehot = tf.one_hot(y_true, depth=y_pred.shape[-1])
        y_pred_softmax = tf.nn.softmax(y_pred, axis=-1)
        epsilon = tf.keras.backend.epsilon()
        y_pred_clipped = tf.clip_by_value(y_pred_softmax, epsilon, 1.0 - epsilon)
        loss = -y_true_onehot * alpha * tf.pow(1 - y_pred_clipped, gamma) * tf.math.log(y_pred_clipped)
        return tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    return focal_loss_fixed


"""
Method to Train the classifier , fine tune the model
"""
def train_bert_classifier(X_train, y_train, X_test, y_test, num_labels, batch_size=32, epochs=5):

    train_dataset = tokenize_data(X_train, y_train)
    test_dataset = tokenize_data(X_test, y_test)

    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Load model and config
    config = BertConfig.from_pretrained('bert-base-uncased', num_labels=num_labels)
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', config=config)

    # Compile model
    optimizer = tf.keras.optimizers.AdamW(learning_rate=5e-5)
    loss = focal_loss(gamma=2.0, alpha=0.25)
    metric = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')

    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True, verbose=1)
    lr_scheduler = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=2, verbose=1, min_lr=1e-6)

    # Train
    finetuned_model = model.fit(train_dataset,validation_data=test_dataset,epochs=epochs,callbacks=[early_stopping, lr_scheduler])

    return model, finetuned_model

""" Method to predict the labels of teh test datset"""

def predict_testdata(model,X_test,y_test, batch_size=32):

    test_dataset = tokenize_data(X_test, y_test)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    probs = model.predict(test_dataset)
    y_predicted = np.argmax(probs.logits, axis=1)
    return y_predicted

"""
Method to plot the confusion matrix i.e, Actual vs Predicted LAbels of the bug dataset
"""
def plot_confusion_matrix(y_test, y_pred, class_labels):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels, cmap="Blues")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

"""
Method to plot the Training Accuracy  vs Validation accuracy and Training loss  vs Validation loss
"""

def plot_training_history(finetuned_model):
    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(finetuned_model.history['accuracy'], label='Train Accuracy')
    plt.plot(finetuned_model.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(finetuned_model.history['loss'], label='Train Loss')
    plt.plot(finetuned_model.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

