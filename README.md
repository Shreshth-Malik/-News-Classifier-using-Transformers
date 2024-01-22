# News Classification using Transformer-based Models

## Overview
This repository showcases our project on news classification using Transformer models like BERT and RoBERTa. Aimed at enhancing efficiency and accuracy in categorizing news articles, this project is a blend of cutting-edge NLP techniques and data analytics.

## Project Description
The project leverages advanced NLP techniques using BERT and RoBERTa models to categorize news articles into specific topics. It demonstrates the effectiveness of Transformer models in understanding context and semantics in text.

## Traditional NLP Challenges
- **Feature Engineering:** Traditional NLP algorithms heavily relied on manual feature engineering. Linguistic experts crafted features like bag-of-words, TF-IDF (Term Frequency-Inverse Document Frequency), and word embeddings to represent text data.
- **Limited Context Understanding:** Earlier models struggled with capturing contextual information of a language. They couldn’t understand the meaning of words in relation to surrounding words, leading to limitations in tasks like sentiment analysis and news classification.

## Key Features
- **Advanced NLP Models:** Utilizing BERT and RoBERTa for deep contextual understanding.
- **Data-Driven Insights:** Thorough exploratory data analysis for insightful model training.
- **Robust Evaluation:** Comparing models using accuracy, precision, recall, and F1-score.

## Why BERT and RoBERTa?
- **BERT's Bidirectional Reading:** BERT's unique feature is its bidirectional reading of text, allowing it to better understand the context of each word. This capability is crucial for tasks like sentiment analysis and question-answering, as it helps in comprehending the nuances and relationships between words in sentences more effectively than traditional models.
- **RoBERTa's Focus on MLM Training:** RoBERTa, an enhancement of BERT, places a stronger emphasis on Masked Language Model (MLM) training. This approach randomly masks some of the words in the input data and then predicts these masked words. This method significantly improves the language understanding capabilities of the model, making it more efficient for complex NLP tasks.

## Data
The dataset, available on Kaggle, comprises 210,000 records spanning the years 2012 to 2022. Notably, it exhibits a slight temporal imbalance, with a significant majority of approximately 200,000 records predating 2018. This dataset encompasses six attributes: category, headline, author, link, short description, and date of publication.

## Methodology
- **Exploratory Data Analysis (EDA):** In the context of news article classification, a thorough EDA helped with gaining insights into the nature of the data, making it easier to preprocess effectively and design a robust model. Various methods used: Text Length Analysis, Examining of Class Distribution, Visualizations, Word Frequency Analysis, etc.
- **Pre-Processing of Data:** Reduced the number of classes in the dataset from 44 to 26 which helps with cutting down redundant classes and increasing the number of records per class. Additionally, since BERT and RoBERTa are neural networks, they require uniform-length input vectors. The BERT and RoBERTa tokenizers in the “transformers” library handle the required padding and truncation during tokenization. We also trained BERT on RoBERTa on both: text that has undergone lemmatization, stopword removal, and punctuation removal (second phase), as well as text that did not (first phase).
- **Training and Testing split:** The distribution of the dataset for this project is organized as follows: the training set comprises 0.7 of the total data, while both the validation set and the test set each constitute 0.15.
- **Model Training:** Rigorously trained the models on the dataset, then fine-tuned the parameters for optimal performance and to suit the needs to suit the needs of news classification.

## Findings
- **Model Comparison:** Evaluated the performance of BERT and RoBERTa models using metrics like accuracy, precision, recall, and F1-score.
- **Insights and Observations:** Detailed analysis of the models' performance both with and without text preprocessing, highlighting the effectiveness of the Transformer-based approach in news classification.
- The RoBERTa model exhibits better performance than the BERT model for the same input data despite a shorter training time. In other words, the RoBERTa model trained on the text that did not undergo text preprocessing does better than the BERT model trained on the same text that did not undergo as well.

## Conclusion
The project demonstrates the effectiveness of Transformer models in NLP, particularly for complex tasks like news classification. It offers insights into the capabilities of BERT and RoBERTa in contextually understanding and categorizing textual content.
