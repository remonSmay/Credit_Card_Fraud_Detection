# import argparse

# from sklearn.model_selection import GridSearchCV
# from credit_fraud_utils_data import load_data, preprocess_data
# from credit_fraud_utils_eval import evaluate_model, find_best_threshold, save_results_to_json
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# import pickle

# def train_model(args):
#     """
#     Main training function.
#     TODO:
#     1. Load data (train, val) using load_data()
#     2. Preprocess data (scaling, sampling) [y] 
#     3. Train Logistic Regression and Random Forest and MLPclassifer
#     4. Evaluate models → get metrics
#     5. Select best models for Voting Classifier
#     6. Find best threshold
#     7. Save model + threshold to pickle
#     8. Save results (metrics + params) to JSON
#     """
#     train_df = load_data(args.data_path)
#     val_df = load_data(args.val_data_path)
#     X_train , y_train =preprocess_data(train_model,args.sampling_strategy, args.sampling_method) # x samplin , y 
#     # the val data is not sampled
#     X_val, y_val = preprocess_data(val_df)
#     # Train Logistic Regression
#     logistic_model = LogisticRegression(max_iter=1000,random_state=42)
#     logistic_model.fit(X_train, y_train)
    
   # train the random forest with grid search

    # rf = RandomForestClassifier(random_state=42, class_weight="balanced")
    # param_grid = {
    # "n_estimators": [100, 200],
    # "max_depth": [None, 10, 20],
    # "min_samples_split": [2, 5, 10],
    # "min_samples_leaf": [1, 2, 4],
    # "max_features": ["sqrt", "log2"]    }

    # grid_search_forest = GridSearchCV(
    #     rf, param_grid, 
    #     scoring="f1",   
    #     cv=3, n_jobs=-1, verbose=2
    # )
    # grid_search_forest.fit(X_train, y_train)
    # rf_best =grid_search_forest.best_estimator_
    # evaluate the models
    # evaluate_model(logistic_model,X_val,y_val)
    # evaluate_model(rf_best,X_val,y_val)
    
   
    



import argparse
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from credit_fraud_utils_data import load_data, preprocess_data
from credit_fraud_utils_eval import evaluate_model, find_best_threshold, save_results_to_json


# ---------------------------
# 1. Data Loading
# ---------------------------
def load_and_preprocess(train_path, val_path, sampling):
    """
    Load and preprocess train/validation data.
    TODO:
    - Load train.csv and val.csv using load_data()
    - Preprocess train (with optional sampling) and val (without sampling)
    - Return X_train, y_train, X_val, y_val, scaler
    """
    
    df_trian = load_data(args.data_path)
    df_val = load_data(args.val_data_path)
    X_train , y_train =preprocess_data(df_trian,args.sampling_strategy, args.sampling_method) # x samplin , y 
    # the val data is not sampled
    X_val, y_val = preprocess_data(df_val)
    
    return X_train,y_train,X_val,y_val

# ---------------------------
# 2. Model Training
# ---------------------------
def train_logistic_regression(X_train, y_train, X_val, y_val):
    """
    Train Logistic Regression model.
    TODO:
    - Initialize LogisticRegression
    - Fit on train set
    - Evaluate on validation set using evaluate_model()
    - Return trained model + metrics dict
    """
    logistic_model=LogisticRegression(max_iter=1000,random_state=42)
    logistic_model.fit(X_train,y_train)
    eval_metric =evaluate_model(X_val,y_val)
    return logistic_model , eval_metric


def train_random_forest(X_train, y_train, X_val, y_val, n_estimators, max_depth):
    """
    Train Random Forest model.
    TODO:
    - Initialize RandomForestClassifier with parameters
    - Fit on train set
    - Evaluate on validation set using evaluate_model()
    - Return trained model + metrics dict
    """
    rf_model=RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, class_weight="balanced")
    pass


def build_voting_classifier(log_reg, rf, X_train, y_train, X_val, y_val):
    """
    Build and train a Voting Classifier from Logistic Regression + Random Forest.
    TODO:
    - Create VotingClassifier with 'soft' voting
    - Fit on train set
    - Evaluate on validation set using evaluate_model()
    - Return trained model + metrics dict
    """
    pass


# ---------------------------
# 3. Model Saving
# ---------------------------
def save_model(model, scaler, threshold, model_path):
    """
    Save final model, scaler, and best threshold to pickle file.
    TODO:
    - Open model_path in write-binary mode
    - Use pickle.dump to store {"model": ..., "scaler": ..., "threshold": ...}
    """
    pass


# ---------------------------
# 4. Orchestrator
# ---------------------------
def train_model(args):
    """
    Main training pipeline.
    TODO:
    1. Load and preprocess train/val data
    2. Train Logistic Regression → get metrics
    3. Train Random Forest → get metrics
    4. Build Voting Classifier → get metrics
    5. Find best threshold on validation
    6. Save final model with scaler + threshold
    7. Save evaluation results (all metrics + params) to JSON
    """
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credit Card Fraud Detection Training")
    # TODO: Add CLI arguments (train_path, val_path, model_path, results_path, sampling, n_estimators, max_depth)
    args = parser.parse_args()

    train_model(args)
