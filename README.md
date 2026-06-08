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

The dataset contains patient-related medical and lifestyle information associated with lung cancer risk prediction.

### Features Included

- Demographic information
- Medical history
- Risk factors
- Clinical indicators

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

### Data Analysis
(Add your visualization screenshots here)

### ROC Curve
(Add ROC Curve image here)

### Confusion Matrix
(Add Confusion Matrix image here)

### GUI Interface
(Add GUI screenshot here)

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
