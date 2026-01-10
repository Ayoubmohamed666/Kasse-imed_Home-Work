# Machine Learning Homework  
**Student:** Kasse Imed Nacer Eddine  
**Level:** 1st Year Master – Artificial Intelligence  

This repository contains two mini-projects implemented in Python using scikit-learn and AutoML tools.

---

## 📌 Project 1: Customer Segmentation (Clustering)

### Objective
Segment customers into groups based on their:
- Age  
- Annual Income  
- Spending Score  

Using the **K-Means** clustering algorithm.

### Dataset
- Source: Kaggle  
- Name: *Mall Customers Dataset*  
- Features:
  - CustomerID  
  - Gender  
  - Age  
  - Annual Income (k$)  
  - Spending Score (1–100)

### Methodology
1. Load dataset using `kagglehub`.
2. Explore data (shape, columns, statistics).
3. Normalize features using `StandardScaler`.
4. Apply K-Means clustering (k = 5).
5. Visualize clusters using scatter plots.

### Interpretation Example
- Cluster 0: High income & high spending → VIP customers  
- Cluster 1: High income & low spending → Target for marketing campaigns  
- Cluster 2: Low income → Discount-oriented customers  
- Other clusters: Medium profiles  

---

## 📌 Project 2: Credit Card Fraud Detection (Classification)

### Objective
Detect fraudulent transactions using supervised learning.

### Dataset
- Source: Kaggle  
- Name: *Credit Card Fraud Dataset*  
- Features:
  - V1 … V28 (PCA-transformed features)
  - Class (0 = normal, 1 = fraud)

### Methodology
1. Load dataset and take a small sample (2%) for fast experiments.
2. Analyze class distribution (highly imbalanced).
3. Split data into Train/Test sets.
4. Normalize features with `StandardScaler`.

### Models Used
- **Logistic Regression**
- **Random Forest**
- **Evaluation Metrics**
  - Classification Report (Precision, Recall, F1-score)
  - Confusion Matrix visualization

### Observations
- The dataset is highly imbalanced.
- Logistic Regression provides a baseline.
- Random Forest generally achieves better recall for fraud detection.

---

## 🤖 AutoML with TPOT

### Objective
Automatically search for the best machine learning pipeline using **TPOT**.

### Configuration
- Generations: 1  
- Population size: 5  
- Parallel jobs: 1 (to avoid Colab timeout issues)

### Output
- Best pipeline is exported as:
