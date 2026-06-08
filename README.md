# 🫁 High-Risk Lung Cancer Patient Classification Using Machine Learning and Deep Learning

## 📌 Project Overview

Lung cancer remains one of the leading causes of cancer-related deaths worldwide. Early identification of high-risk patients can significantly improve treatment outcomes and survival rates.

This project focuses on developing and comparing Machine Learning and Deep Learning models for the classification of high-risk lung cancer patients. The study includes a complete end-to-end machine learning pipeline, from data preprocessing and class balancing to model evaluation and GUI implementation.

---

## 🎯 Objectives

- Identify patients with high lung cancer risk.
- Compare traditional Machine Learning algorithms with Deep Learning approaches.
- Handle class imbalance using SMOTE.
- Evaluate model performance using multiple metrics.
- Develop an interactive prediction interface.

---

## 📊 Dataset

The dataset used in this project was obtained from Kaggle and contains clinical, demographic, and health-related information associated with lung cancer patients.

The objective of the dataset is to support the classification of diagnosed lung cancer patients into Low-Risk and High-Risk categories.

### Dataset Source

Kaggle Dataset:
[(https://www.kaggle.com/datasets/rashadrmammadov/lung-cancer-prediction)]
### Features Included

- Demographic Information
- Medical History
- Clinical Indicators
- Lifestyle Factors
- Risk Assessment Variables
---

## ⚙️ Data Preprocessing

Before training the models, several preprocessing techniques were applied:

- Missing value handling
- Feature encoding
- Feature scaling and normalization
- Outlier inspection
- Train-Test splitting
- SMOTE (Synthetic Minority Oversampling Technique)

---

## 🤖 Machine Learning Models

The following Machine Learning algorithms were implemented and evaluated:

### 1. Logistic Regression
A statistical classification model commonly used for binary classification tasks.

### 2. K-Nearest Neighbors (KNN)
Classifies samples based on the nearest neighboring observations.

### 3. Decision Tree
A tree-based model that learns decision rules from data features.

### 4. Random Forest
An ensemble learning technique that combines multiple decision trees for improved accuracy and robustness.

---

## 🧠 Deep Learning Model

### Artificial Neural Network (ANN)

A fully connected neural network was developed using TensorFlow/Keras to learn complex patterns within the dataset.

Architecture included:

- Input Layer
- Hidden Layers
- ReLU Activation Functions
- Output Layer with Sigmoid Activation

---

## 📈 Performance Evaluation

Model performance was evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix

---

## 📊 Visualizations

The project includes various visual analyses such as:

- Data Distribution Plots
- Correlation Heatmaps
- Feature Analysis
- ROC Curves
- Confusion Matrices
- Model Comparison Charts

---

## 🏆 Results

After comparing all models, the best-performing model achieved strong classification performance for identifying high-risk lung cancer patients.

Key findings:

- SMOTE improved minority class prediction performance.
- Ensemble methods demonstrated strong classification capability.
- Deep Learning successfully captured complex feature relationships.
- ROC-AUC analysis provided reliable model comparison.

---

## 💻 GUI Application

A graphical user interface (GUI) was developed to allow users to input patient information and receive predictions in an intuitive way.

Features:

- User-friendly design
- Real-time prediction
- Easy data entry
- Model integration

---

## 🛠️ Technologies Used

| Category | Technologies |
|-----------|-------------|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-Learn |
| Deep Learning | TensorFlow, Keras |
| Data Balancing | SMOTE (Imbalanced-Learn) |
| GUI Development | Tkinter |
| Version Control | Git, GitHub |

---

## 📂 Project Structure

```text
High-Risk-Lung-Cancer-Patient-Classification/
│
├── Dataset/
├── Notebooks/
├── GUI/
├── Reports/
├── Models/
├── Results/
└── README.md
```

---
## 📸 Project Screenshots

### Correlation Matrix

![Correlation Matrix](High-Risk%20Lung%20Cancer%20Patient%20Classification%20Using%20Machine%20Learning%20and%20Deep%20Learning/04_Output_Figures/Correlation_Matrix.png)

### Feature Distribution

![Feature Distribution](High-Risk%20Lung%20Cancer%20Patient%20Classification%20Using%20Machine%20Learning%20and%20Deep%20Learning/04_Output_Figures/Feature_Distribution.png)

### Feature Importance

![Feature Importance](High-Risk%20Lung%20Cancer%20Patient%20Classification%20Using%20Machine%20Learning%20and%20Deep%20Learning/04_Output_Figures/Feature_Importance.png)

### ROC Curve

![ROC Curve](High-Risk%20Lung%20Cancer%20Patient%20Classification%20Using%20Machine%20Learning%20and%20Deep%20Learning/04_Output_Figures/ROC_Curve.png)

### Confusion Matrix

![Confusion Matrix](High-Risk%20Lung%20Cancer%20Patient%20Classification%20Using%20Machine%20Learning%20and%20Deep%20Learning/04_Output_Figures/Confusion_Matrix.png)

### Model Accuracy Comparison

![Model Accuracy Comparison](High-Risk%20Lung%20Cancer%20Patient%20Classification%20Using%20Machine%20Learning%20and%20Deep%20Learning/04_Output_Figures/Model_Accuracy_Comparison.png)

### GUI Interface

![GUI Interface](High-Risk%20Lung%20Cancer%20Patient%20Classification%20Using%20Machine%20Learning%20and%20Deep%20Learning/04_Output_Figures/GUI.png)

---

## 🚀 Future Improvements

- Hyperparameter optimization
- Additional ensemble models
- Larger datasets
- Explainable AI (XAI) integration
- Web-based deployment

---

## 👨‍💻 Author

**Polat Bozuklu**

Computer Engineering Student

---

## 📜 License

This project was developed for educational and research purposes as part of a Machine Learning course project.
