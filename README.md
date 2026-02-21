# 📧 Spam Email Detection using TensorFlow (LSTM)

This project builds a **Spam Email Detection System** using **Deep Learning (LSTM)** with **TensorFlow/Keras**.

The model classifies emails into:

* ❌ **Spam**
* ✅ **Ham (Not Spam)**

The system automatically detects unwanted or unsolicited emails, helping reduce inbox clutter.



## 🚀 Project Overview

Spam detection is a classic **Natural Language Processing (NLP)** problem.

In this project,I:

* Clean and preprocess text data
* Balance the dataset
* Convert text into numerical sequences
* Train a deep learning model using **LSTM**
* Achieve ~97% accuracy


## 🛠️ Technologies & Libraries Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* NLTK
* WordCloud
* TensorFlow / Keras
* Scikit-learn



## 📂 Dataset

The dataset (`Emails.csv`) contains:

* `text` → Email content
* `label` → Spam or Ham

Dataset size:

* 📊 **5171 rows**
* 4 columns

Initially, the dataset is **imbalanced** (Ham > Spam), so we balance it using **downsampling**.



###  Model Architecture

We build a **Sequential LSTM model**:

| Layer           | Purpose                      |
| --------------- | ---------------------------- |
| Embedding       | Learns word representations  |
| LSTM            | Captures sequential patterns |
| Dense (ReLU)    | Feature extraction           |
| Dense (Sigmoid) | Binary classification        |


### Model Performance

✅ **Test Accuracy: 97%**
📉 **Test Loss: 0.1202**

This indicates strong spam detection capability.

## 🧠 Key Concepts Used

* Natural Language Processing (NLP)
* Tokenization
* Padding
* Word Embeddings
* LSTM (Long Short-Term Memory)
* Binary Classification
* Class Imbalance Handling
* Early Stopping


## 🎯 Future Improvements

* Use Bidirectional LSTM
* Add Dropout layers
* Try GRU instead of LSTM
* Use pre-trained embeddings (GloVe/Word2Vec)
* Deploy using Streamlit
* Convert into a Web App

