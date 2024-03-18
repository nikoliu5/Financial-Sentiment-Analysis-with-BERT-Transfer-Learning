# Financial-Sentiment-Analysis-with-BERT-Transfer-Learning
This project utilizes BERT transfer learning to classify sentiment in financial texts

<img src="./src_img/sentiment_analysis.jpg" width="650">

## Project Overview
This project focuses on performing sentiment analysis on sentences from the financial domain using transfer learning with the BERT (Bidirectional Encoder Representations from Transformers) model. The objective is to develop a model that can accurately classify the sentiment of financial phrases as Negative, Neutral, or Positive.

## Data
The Financial PhraseBank dataset, a collection of hand-annotated sentences with sentiment labels, is utilized in this project. The dataset offers various "flavors" or subsets based on the level of agreement among annotators. For the basic part of the project, the "all agree" flavor is used.

[Financial PhraseBank - Hugging Face](https://huggingface.co/datasets/financial_phrasebank)

## Transfer Learning Approach
The project leverages transfer learning, which involves using a pre-trained BERT model and adapting it to the specific task of financial sentiment analysis. The main steps include:

Loading a pre-trained BERT model from the HuggingFace platform.
Freezing certain layers of the BERT model to retain the learned representations.
Adding a custom classification head on top of the frozen BERT layers.
Fine-tuning the model on the Financial PhraseBank dataset.
By freezing layers of the BERT model, the project aims to utilize the powerful language understanding capabilities of BERT while reducing the computational overhead and preventing overfitting to the specific domain.


## Project Outline
```
Financial Sentiment Analysis with BERT Transfer Learning
|
├── Initial Setup
| ├── Installation of Required Packages
| ├── GPU Availability Check
| └── TensorFlow Version Verification
|
├── Modeling Pipeline (Part I)
| ├── Data Acquisition
| | └── Importing HuggingFace Dataset
| ├── Data Analysis
| | └── Exploratory Data Analysis
| ├── Data Preparation
| | ├── Transformation
| | ├── Tokenization
| | └── Addressing Imbalanced Data Issues
| └── Model Training
| ├── Training the Classifier Head
| ├── Training All Layers
| └── Error Analysis
|
└── Performance Enhancement (Part II)
├── Advanced Data Handling
| └── TensorFlow Dataset (TFDS) Integration
├── Data Imbalance
├── Model Architecture
| └── Custom Classification Head Design
├── In-Depth Error Analysis
├── Dataset Variation
| └── Utilizing Different Dataset "Flavors"
├── Pre-Trained Model Experimentation
| └── Testing Various Pre-Trained Models
└── Fine-Tuning
└── Experiment with Few-shot Learning
```

### Analysis and Results
Our model achieves notable performance in classifying sentiments of financial texts, demonstrating the effectiveness of transfer learning with BERT for this domain. Detailed results and performance metrics are presented in the notebook.

## Requirements
Please refer to `requirements.txt` for a complete list of dependencies.