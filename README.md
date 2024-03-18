# Financial-Sentiment-Analysis-with-BERT-Transfer-Learning
This project utilizes BERT transfer learning to classify sentiment in financial texts


This project explores the application of BERT (Bidirectional Encoder Representations from Transformers) for sentiment analysis in financial texts. By leveraging transfer learning, we fine-tune a pre-trained BERT model to classify the sentiment of financial news articles and reports as positive, neutral, or negative. This repository contains a Jupyter notebook that guides you through the process of data preparation, model training, and evaluation.

<img src="./src_img/sentiment_analysis.jpg" width="650">

## Project Overview

- **Objective**: To apply BERT for accurate sentiment classification of financial texts.
- **Data Source**: Utilize a curated dataset of financial news articles from various sources.
- **Approach**: Employ transfer learning techniques with a pre-trained BERT model.

## Data

The dataset comprises financial news articles collected from multiple sources, labeled with sentiments as positive, negative, or neutral. We split the dataset into training, validation, and testing parts to ensure a robust evaluation of the model's performance.

## Model Training and Evaluation


```
BERT Model Training
|
├── Data Preprocessing
|   |
|   └── Tokenization, Padding, and Masking
|
├── Model Fine-tuning
|   |
|   └── Training with Custom Dataset
|
└── Evaluation
    |
    ├── Accuracy, Precision, Recall, and F1-Score
```

### Transfer Learning with BERT

- **Fine-tuning**: We adapt the pre-trained BERT model to our specific task of sentiment analysis.
- **Training Strategy**: Utilize AdamW optimizer with a learning rate schedule that includes a warm-up phase.

### Results

Our model achieves notable performance in classifying sentiments of financial texts, demonstrating the effectiveness of transfer learning with BERT for this domain. Detailed results and performance metrics are presented in the notebook.

## Requirements

To run the notebook, you will need the following libraries:

- transformers
- torch
- numpy
- pandas
- matplotlib

Please refer to `requirements.txt` for a complete list of dependencies.