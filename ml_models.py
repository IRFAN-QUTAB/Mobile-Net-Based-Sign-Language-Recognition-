#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


# In[ ]:



X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

y_train = np.argmax(y_train, axis=1)
y_test = np.argmax(y_test, axis=1)

print(f"Loaded feature shapes: Train {X_train.shape}, Test {X_test.shape}")
    

### 📌 Train and Evaluate Each Model Using Extracted Features

# K-Nearest Neighbors (KNN)
KNN = KNeighborsClassifier(n_neighbors=5)
KNN.fit(X_train, y_train)
y_pred_knn = KNN.predict(X_test)
display_results(KNN, y_pred_knn, "KNN")

def display_results(model, y_pred, KNN):
    # Compute Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Print formatted metrics
    print(f"\n{KNN} Classification Report:\n")
    print(f"Accuracy = {accuracy:.16f}")
    print(f"F1-Score {f1:.16f}")
    print(f"Recall {recall:.16f}")
    print(f"Precision {precision:.16f}\n")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[chr(i) for i in range(65, 91)], yticklabels=[chr(i) for i in range(65, 91)])
    plt.title(f"{KNN} - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.show()

# Logistic Regression (LR) - Best Performing Model
LR = LogisticRegression(
    max_iter=2000,
    solver='lbfgs',  
    C=1.0,            
    penalty='l2',     
    multi_class='multinomial',  
    class_weight='balanced'
)
LR.fit(X_train, y_train)
y_pred_lr = LR.predict(X_test)
display_results(LR, y_pred_lr, "Logistic Regression")

def display_results(model, y_pred,KNN):
    # Compute Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Print formatted metrics
    print(f"\n{LR} Classification Report:\n")
    print(f"Accuracy = {accuracy:.16f}")
    print(f"F1-Score {f1:.16f}")
    print(f"Recall {recall:.16f}")
    print(f"Precision {precision:.16f}\n")
    print(classification_report(y_test, y_pred))
    
    # Generate Confusion Matrix (Styled as Per Your Image)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[chr(i) for i in range(65, 91)], yticklabels=[chr(i) for i in range(65, 91)])
    plt.title(f"{LR} - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.show()
    
# Random Forest (RF)
RF = RandomForestClassifier(n_estimators=100, random_state=42)
RF.fit(X_train, y_train)
y_pred_rf = RF.predict(X_test)
display_results(RF, y_pred_rf, "Random Forest")

    # Print formatted metrics
    print(f"\n{RF} Classification Report:\n")
    print(f"Accuracy = {accuracy:.16f}")
    print(f"F1-Score {f1:.16f}")
    print(f"Recall {recall:.16f}")
    print(f"Precision {precision:.16f}\n")
    print(classification_report(y_test, y_pred))
    
    # Generate Confusion Matrix (Styled as Per Your Image)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[chr(i) for i in range(65, 91)], yticklabels=[chr(i) for i in range(65, 91)])
    plt.title(f"{RF} - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.show()
    
    
# Naive Bayes (NB)
NB = GaussianNB()
NB.fit(X_train, y_train)
y_pred_nb = NB.predict(X_test)
display_results(NB, y_pred_nb, "Naive Bayes")

    # Print formatted metrics
    print(f"\n{NB} Classification Report:\n")
    print(f"Accuracy = {accuracy:.16f}")
    print(f"F1-Score {f1:.16f}")
    print(f"Recall {recall:.16f}")
    print(f"Precision {precision:.16f}\n")
    print(classification_report(y_test, y_pred))
    
    # Generate Confusion Matrix (Styled as Per Your Image)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[chr(i) for i in range(65, 91)], yticklabels=[chr(i) for i in range(65, 91)])
    plt.title(f"{NB} - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.show()

# Decision Tree (DT)
DT = DecisionTreeClassifier(random_state=42)
DT.fit(X_train, y_train)
y_pred_dt = DT.predict(X_test)
display_results(DT, y_pred_dt, "Decision Tree")

    # Print formatted metrics
    print(f"\n{DT} Classification Report:\n")
    print(f"Accuracy = {accuracy:.16f}")
    print(f"F1-Score {f1:.16f}")
    print(f"Recall {recall:.16f}")
    print(f"Precision {precision:.16f}\n")
    print(classification_report(y_test, y_pred))
    
    # Generate Confusion Matrix (Styled as Per Your Image)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[chr(i) for i in range(65, 91)], yticklabels=[chr(i) for i in range(65, 91)])
    plt.title(f"{DT} - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.show()


# In[ ]:




