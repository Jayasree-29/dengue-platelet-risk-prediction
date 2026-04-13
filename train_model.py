import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, mean_absolute_error, mean_squared_error

def train_models():
    print("Step 1: Data Preprocessing...")
    # Load dataset
    data_path = os.path.join(os.path.dirname(__file__), 'Dengue-Dataset.csv')
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return

    df = pd.read_csv(data_path)

    # Clean column names
    df.columns = [c.split('(')[0].strip() for c in df.columns]
    
    # 1.1 Handle Missing Values
    # Identify numerical columns
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    # Identify categorical columns
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove Target variable from features list if present
    if 'Result' in categorical_features:
        categorical_features.remove('Result')
    
    # Impute numerical with median
    imputer_num = SimpleImputer(strategy='median')
    df[numeric_features] = imputer_num.fit_transform(df[numeric_features])
    
    # 1.2 Encode Gender (Male/Female -> 0/1)
    le_gender = LabelEncoder()
    if 'Gender' in df.columns:
        df['Gender'] = le_gender.fit_transform(df['Gender'].astype(str))
    
    # Encode Result (Target)
    le_result = LabelEncoder()
    df['Result'] = le_result.fit_transform(df['Result'].astype(str).str.lower())
    
    # 1.3 Outlier Removal (Interquartile Range Method)
    print("Step 1.3: Removing Outliers...")
    numeric_for_outliers = [col for col in numeric_features if col in df.columns]
    Q1 = df[numeric_for_outliers].quantile(0.25)
    Q3 = df[numeric_for_outliers].quantile(0.75)
    IQR = Q3 - Q1
    
    initial_shape = df.shape[0]
    df = df[~((df[numeric_for_outliers] < (Q1 - 1.5 * IQR)) | (df[numeric_for_outliers] > (Q3 + 1.5 * IQR))).any(axis=1)]
    print(f"Removed {initial_shape - df.shape[0]} outliers. New shape: {df.shape[0]}")

    print("Step 2: Feature Engineering...")
    # 2.1 Feature Scaling
    # We scale specific features that vary widely in range (e.g., Platelet Count vs Hemoglobin)
    features_to_scale = ['Age', 'Hemoglobin', 'RBC', 'HCT', 'MCV', 'MCH', 'MCHC', 
                         'RDW-CV', 'Total Platelet Count', 'MPV', 'PDW', 'PCT', 'Total WBC count']
    
    # Intersection of what we want to scale and what exists
    scale_cols = [col for col in features_to_scale if col in df.columns]
    
    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])
    
    # Define X and y
    X = df.drop(['Result'], axis=1)
    y = df['Result']
    
    print(f"Training with features: {list(X.columns)}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Step 3: ML Model Processing...")
    
    
    # 3.1 Logistic Regression (Baseline)
    print("\n--- Logistic Regression ---")
    lr_model = LogisticRegression(max_iter=2000, random_state=42)
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
    print(classification_report(y_test, y_pred_lr))
    
    # 3.2 Random Forest Classifier
    print("\n--- Random Forest Classifier ---")
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
    print(classification_report(y_test, y_pred_rf))
    
    # 3.3 XGBoost Classifier
    print("\n--- XGBoost Classifier ---")
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
    print(classification_report(y_test, y_pred_xgb))
    
    # 3.4 XGBoost Regressor for Platelet Decline Forecast
    print("\n--- XGBoost Regressor (Decline Forecast) ---")
    
    X_reg = df.drop(['Total Platelet Count', 'Result'], axis=1)
    y_reg = df['Total Platelet Count'] # This is currently Scaled
    
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    
    reg_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    reg_model.fit(X_train_reg, y_train_reg)
    
    y_pred_reg = reg_model.predict(X_test_reg)
    mae = mean_absolute_error(y_test_reg, y_pred_reg)
    rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
    
    print(f"Mean Absolute Error (Scaled): {mae:.4f}")
    print(f"Root Mean Squared Error (Scaled): {rmse:.4f}")
    # Save artifacts
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    joblib.dump(lr_model, os.path.join(models_dir, 'lr_model.pkl'))
    joblib.dump(rf_model, os.path.join(models_dir, 'rf_model.pkl'))
    joblib.dump(xgb_model, os.path.join(models_dir, 'xgb_model.pkl'))
    joblib.dump(reg_model, os.path.join(models_dir, 'reg_model.pkl'))
    
    joblib.dump(imputer_num, os.path.join(models_dir, 'imputer.pkl'))
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    joblib.dump(le_gender, os.path.join(models_dir, 'le_gender.pkl'))
    joblib.dump(le_result, os.path.join(models_dir, 'le_result.pkl'))
    
    # Save feature lists to ensure order during inference
    joblib.dump(X.columns.tolist(), os.path.join(models_dir, 'feature_names.pkl'))
    joblib.dump(X_reg.columns.tolist(), os.path.join(models_dir, 'reg_feature_names.pkl'))
    joblib.dump(scale_cols, os.path.join(models_dir, 'scale_cols.pkl'))
    
    print("Models and preprocessing artifacts saved successfully.")

    # Create results folder
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Save Accuracy Report
    report_path = os.path.join(results_dir, 'accuracy_report.txt')
    with open(report_path, 'w') as f:
        f.write("Dengue Platelet Guardian - Accuracy Report\n")
        f.write("==========================================\n\n")
        f.write(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}\n")
        f.write(f"XGBoost Classifier Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}\n")
        f.write(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}\n\n")
        
        f.write("--- Random Forest Classification Report ---\n")
        f.write(classification_report(y_test, y_pred_rf))
        
        f.write("\n--- XGBoost Classification Report ---\n")
        f.write(classification_report(y_test, y_pred_xgb))
        
        f.write("\n--- Logistic Regression Classification Report ---\n")
        f.write(classification_report(y_test, y_pred_lr))
        
        f.write(f"\nPlatelet Forecast MAE (Scaled): {mae:.4f}\n")
    
    # Generate Plots
    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred_rf)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le_result.classes_, yticklabels=le_result.classes_)
    plt.title('Dengue Prediction Confusion Matrix (Random Forest)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
    plt.close()

    # 2. Feature Importance
    plt.figure(figsize=(10, 8))
    importances = rf_model.feature_importances_
    feat_importances = pd.Series(importances, index=X.columns)
    feat_importances.nlargest(10).plot(kind='barh', color='skyblue')
    plt.title('Top 10 Features Impacting Prediction')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'feature_importance.png'))
    plt.close()

    print(f"Results and plots saved successfully in: {results_dir}")

if __name__ == "__main__":
    train_models()
