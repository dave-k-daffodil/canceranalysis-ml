import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
def load_data():
    try:
        X = pd.read_csv('dataset/data.csv', index_col=0)
        y_df = pd.read_csv('dataset/labels.csv', index_col=0)
        y = y_df.iloc[:, 0]
        print(f"Data loaded successfully: {X.shape[0]} samples, {X.shape[1]} features.")
    except FileNotFoundError:
        print("Dataset not found. Generating dummy data matching the problem dimensionality for demonstration...")
        X = pd.DataFrame(np.random.normal(0, 1, size=(881, 20531)), columns=[f'gene_{i}' for i in range(20531)])
        y = pd.Series(np.random.choice(['BRCA', 'KIRC', 'COAD', 'LUAD', 'PRAD'], size=881), name='Class')
    return X, y

X, y = load_data()
missing_counts = X.isnull().sum().sum()
print(f"Total missing values: {missing_counts}")
if missing_counts > 0:
    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
classes = label_encoder.classes_
print("Target Classes:", classes)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
print("Standard scaling applied.")
var_thresh = VarianceThreshold(threshold=0.1)
X_train_var = var_thresh.fit_transform(X_train_scaled)
X_test_var = var_thresh.transform(X_test_scaled)

kept_features = X_train_scaled.columns[var_thresh.get_support()]
X_train_var = pd.DataFrame(X_train_var, columns=kept_features)
X_test_var = pd.DataFrame(X_test_var, columns=kept_features)
print(f"Features retained after Variance Threshold: {X_train_var.shape[1]}")

k_features = min(500, X_train_var.shape[1])
selector = SelectKBest(score_func=f_classif, k=k_features)
X_train_selected = pd.DataFrame(selector.fit_transform(X_train_var, y_train), columns=kept_features[selector.get_support()])
X_test_selected = pd.DataFrame(selector.transform(X_test_var), columns=kept_features[selector.get_support()])
print(f"Final features shape for models post ANOVA: {X_train_selected.shape}")

# Bonus: Applying PCA as a holistic data reduction visualizer
pca = PCA(n_components=min(100, X_train_scaled.shape[1]), random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
print(f"Variation captured by Top 100 PCA Components: {np.sum(pca.explained_variance_ratio_):.4f}")
models = {
    'Linear_SVM': SVC(kernel='linear', probability=True, random_state=42),
    'Random_Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(eval_metric='mlogloss', random_state=42)
}

param_grids = {
    'Linear_SVM': {'C': [0.01, 0.1, 1, 10]},
    'Random_Forest': {'n_estimators': [50, 150], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]},
    'XGBoost': {'n_estimators': [50, 150], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]}
}

best_models = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\nTuning {name}...")
    grid = RandomizedSearchCV(model, param_grids[name], cv=cv, scoring='accuracy', n_iter=5, random_state=42, n_jobs=-1)
    grid.fit(X_train_selected, y_train)
    best_models[name] = grid.best_estimator_
    print(f"Optimized Parameters: {grid.best_params_}")
    
    # Evaluates generalizability, mitigating overfitting gaps
    cv_res = cross_validate(grid.best_estimator_, X_train_selected, y_train, cv=cv, return_train_score=True, scoring='accuracy')
    print(f"CV Train Accuracy: {cv_res['train_score'].mean():.4f} | Validation CV Accuracy: {cv_res['test_score'].mean():.4f}")
best_model_name = max(best_models, key=lambda k: cross_validate(best_models[k], X_train_selected, y_train, cv=cv, scoring='accuracy')['test_score'].mean())
final_model = best_models[best_model_name]
print(f"\nSelected Core Framework: {best_model_name}")

y_pred = final_model.predict(X_test_selected)

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=classes))

plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title(f'Confusion Matrix Evaluation ({best_model_name})')
plt.ylabel('Actual Category')
plt.xlabel('Predicted Category')
plt.show()
def explain_model(model, name, features):
    if name in ['Random_Forest', 'XGBoost']:
        m_importances = model.feature_importances_
    elif name == 'Linear_SVM':
        m_importances = np.abs(model.coef_).mean(axis=0)
    else:
        return
        
    indices = np.argsort(m_importances)[::-1][:20]
    t_genes = features[indices]
    t_scores = m_importances[indices]
    
    plt.figure(figsize=(10,6))
    plt.barh(range(20), t_scores[::-1], align='center', color='indigo')
    plt.yticks(range(20), t_genes[::-1])
    plt.title(f'Top 20 Critical Gene Discriminants ({name})')
    plt.xlabel('Algorithmic Importance')
    plt.tight_layout()
    plt.show()
    
    print("Biological Context: Identifying the highest influence variations map significant diagnostic power, tracing tumor mutation signatures that isolate physical behaviors separating different forms of genetic cancers.")

explain_model(final_model, best_model_name, X_train_selected.columns)