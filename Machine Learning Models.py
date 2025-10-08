from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score

# Define classifiers - Extended set
classifiers = {
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Naive Bayes': GaussianNB(),
    'MLP': MLPClassifier(max_iter=1000, random_state=42)
}

# Define parameter grids for grid search
param_grids = {
    'Logistic Regression': { 'C': [0.01, 0.1, 1, 10], 
                             'solver': ['lbfgs', 'liblinear']
},
    'SVM': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    },
    'Random Forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    },
    'KNN': {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    },
    'Decision Tree': {
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'criterion': ['gini', 'entropy']
    },
    'Naive Bayes': {
        'var_smoothing': [1e-9, 1e-8, 1e-7]
    },
    'MLP': {
        'hidden_layer_sizes': [(100,), (200,), (100, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001],
        'learning_rate_init': [0.001, 0.01]
    }
}


print("7 Classifiers configured with parameter grids for optimization!")
print(f"Models: {list(classifiers.keys())}")

# Store results
results = {}
best_models = {}

# Train each classifier
for clf_name, clf in classifiers.items():
    print(f"\n{'='*50}")
    print(f"Training {clf_name}")
    print('='*50)
    
    # Grid search
    grid_search = GridSearchCV(
        estimator=clf,
        param_grid=param_grids[clf_name],
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the model
    grid_search.fit(X_train_features, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    best_models[clf_name] = best_model
    
    # Assign for saving
    best_model_Ausl = best_model  
    
    # Make predictions
    y_pred = best_model.predict(X_test_features)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # ROC-AUC (if classifier supports probability prediction)
    try:
        y_proba = best_model.predict_proba(X_test_features)
        roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
    except AttributeError:
        roc_auc = None  # Some models (like SVM without probability=True) don't support predict_proba
    
    # Store results
    results[clf_name] = {
        'model': best_model,
        'best_params': grid_search.best_params_,
        'cv_score': grid_search.best_score_,
        'test_accuracy': accuracy,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }
    
    # Save the model
    filename = f"model_{clf_name.replace(' ', '_').lower()}_ausl.pkl"
    joblib.dump(best_model_Ausl, filename)
    print(f"✓ Model saved as {filename}")

# ================================
# Convert results to DataFrame
# ================================
metrics_df = pd.DataFrame.from_dict(results, orient='index')
metrics_df = metrics_df.drop(columns=['model', 'predictions'], errors='ignore')  # drop non-numeric cols
print("\n📊 Model Comparison Table:\n")
print(metrics_df.round(4))







