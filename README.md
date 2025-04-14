# Federated Learning for Detecting Harmful Content and Disinformation on Twitter

This repository contains the implementation of a Federated Learning (FL) framework for detecting hate speech and fake news on social media platforms, specifically Twitter. The project is based on my Bachelor Thesis at Cyprus University of Technology.

## Project Overview

The goal of this project is to build a **privacy-preserving text classification model** capable of identifying harmful content such as offensive language and disinformation, using **Federated Learning** with **TensorFlow Federated (TFF)**.

The model:
- Uses **GRU-based neural networks** for text classification.
- Trains in a **federated setting**, simulating non-IID data distributions (label imbalance, quantity skew).
- Evaluates performance across multiple datasets related to offensive language and misinformation.

## Key Features

- **Privacy-Preserving Learning** using FL
- **Natural Language Processing (NLP)** for preprocessing tweets
- Simulated **non-IID data partitioning**
- Performance evaluation with **accuracy, F1 score, AUC**, and more
- Graph generation and LaTeX summaries for analysis

## Datasets Used

- Abusive Tweets Dataset
- Offensive Tweets Dataset
- Hateful Tweets Dataset
- Sarcastic Tweets Dataset
- MM-COVID (Fake News) Dataset
