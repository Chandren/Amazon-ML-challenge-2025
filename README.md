# Amazon-ML-challenge-2025
Multi-modal machine learning solution developed for the Amazon ML Challenge 2025 (Unstop), using text and image features with ensemble models.
# Amazon ML Challenge 2025 – Multi-Modal Machine Learning Solution

This repository contains our solution for the **Amazon ML Challenge 2025**, a student machine learning competition hosted on the **Unstop** platform.  
The challenge focused on solving a real-world prediction problem using large-scale, noisy, multi-modal data provided as part of the competition.

Our team participated as a group of three, and this repository reflects the **end-to-end machine learning pipeline** I primarily worked on, including data processing, feature engineering, model training, ensembling, and prediction generation.

---

##  Problem Overview

The task involved building a predictive model on a dataset of approximately **75,000 samples**, where each item contained:
- **Textual information** (product metadata and descriptions)
- **Image information** (product images via URLs)

The goal was to predict a target value (price-related) under real-world constraints such as:
- missing or broken image links,
- skewed target distribution,
- noisy text fields,
- and limited time for experimentation.

---

##  Approach & Methodology

To address the multi-modal nature of the problem, we designed a pipeline that combines **text-based** and **image-based** features, followed by ensemble learning.

### 1. Text Features
- Hand-crafted statistical features from product text
- Semantic embeddings using **Sentence-BERT (all-MiniLM-L6-v2)**

### 2. Image Features
- Image downloads from URLs with robust error handling
- Deep feature extraction using **ResNet-50** (pre-trained on ImageNet)
- Additional image statistics (dimensions, brightness, aspect ratio)

### 3. Model Training
- **XGBoost** for strong non-linear learning
- **LightGBM** for efficient gradient boosting
- Validation using an **80/20 train–validation split**

### 4. Ensembling
- Weighted ensemble of XGBoost and LightGBM
- Weights optimized based on validation performance

---

##  Results

- **Final validation performance:** ~**38% SMAPE**
- Generated predictions for all **75,000 test samples**
- Output validated for:
  - correct shape and schema,
  - no missing or invalid values,
  - positive target predictions.

While the performance could not be pushed further within the competition timeline, the project provided **strong hands-on experience** in:
- large-scale feature engineering,
- multi-modal ML pipelines,
- ensemble modeling,
- and robust ML system design under constraints.

---

##  Challenges faced during the process

- Partial image download failures due to broken or unavailable URLs
- Highly skewed target distribution
- Limited time for extensive hyperparameter tuning
- Computational constraints when handling large feature matrices

These challenges closely mirror real-world ML problems and influenced design and trade-offs throughout the project.



