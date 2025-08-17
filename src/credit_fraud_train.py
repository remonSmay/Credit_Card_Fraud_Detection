
import argparse
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from credit_fraud_utils_data import load_data, preprocess_data
from credit_fraud_utils_eval import evaluate_model, find_best_threshold, save_results_to_json
from sklearn.neural_network import MLPClassifier


def load_and_preprocess(train_path, val_path, sampling):
    """
    Load and preprocess train/validation data.
    TODO:
    - Load train.csv and val.csv using load_data()
    - Preprocess train (with optional sampling) and val (without sampling)
    - Return X_train, y_train, X_val, y_val, scaler
    """
    
    df_train = load_data(train_path)
    df_val = load_data(val_path)
    
    # Convert sampling string to appropriate format
    sampling_strategy = None
    sampling_method = None
    
    if sampling == 'smote':
        sampling_strategy = 0.5  # Balance the classes
        sampling_method = 'smote'
    elif sampling == 'undersampling':
        sampling_strategy = 0.5  # Balance the classes
        sampling_method = 'undersampling'
    elif sampling == 'none':
        sampling_strategy = None
        sampling_method = None
    
    # Preprocess training data with sampling
    X_train, y_train = preprocess_data(df_train, sampling_strategy, sampling_method)
    
    # Preprocess validation data without sampling
    X_val, y_val = preprocess_data(df_val)
    
    # Create a scaler for later use (we'll need to refactor preprocess_data to return scaler)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(df_train.drop('Class', axis=1))
    
    return X_train, y_train, X_val, y_val, scaler

def train_logistic_regression(X_train, y_train, X_val, y_val):
    """
    Train Logistic Regression model.
    TODO:
    - Initialize LogisticRegression
    - Fit on train set
    - Evaluate on validation set using evaluate_model()
    - Return trained model + metrics dict
    """
    logistic_model=LogisticRegression(max_iter=1000,random_state=42, class_weight="balanced", solver='liblinear')
    logistic_model.fit(X_train,y_train)
    metrics_dict =evaluate_model(logistic_model,X_val,y_val)
    return logistic_model , metrics_dict


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
    rf_model.fit(X_train, y_train)
    metrics_dict = evaluate_model(rf_model, X_val, y_val)
    return rf_model , metrics_dict

def train_MLPClassifier(X_train, y_train, X_val, y_val):
    """
    Train MLPClassifier model. 
    TODO:
    - Initialize MLPClassifier
    - Fit on train set
    - Evaluate on validation set using evaluate_model()
    - Return trained model + metrics dict
    """
    
    Mlp_Classifier = MLPClassifier(hidden_layer_sizes=(16,32),
                                   max_iter=1000,
                                   random_state=42,
                                   early_stopping=True, 
                                   solver='adam' ,
                                   validation_fraction=0.1)
    
    Mlp_Classifier.fit(X_train, y_train)
    metrics_dict = evaluate_model(Mlp_Classifier, X_val, y_val)
    
    return Mlp_Classifier, metrics_dict



def build_voting_classifier(log_reg, rf,mlp, X_train, y_train, X_val, y_val):
    """
    Build and train a Voting Classifier from Logistic Regression, Random Forest, and MLPClassifier.
    TODO:
    - Create VotingClassifier with 'soft' voting
    - Fit on train set
    - Evaluate on validation set using evaluate_model()
    - Return trained model + metrics dict
    """
    voting_clf = VotingClassifier(estimators=[('log_reg', log_reg), ('rf', rf), ('mlp', mlp)], voting='soft', weights=None) 
    voting_clf.fit(X_train, y_train)
    metrics_dict = evaluate_model(voting_clf, X_val, y_val)
    return voting_clf, metrics_dict



def save_model(model, scaler, threshold, model_path):
    """
    Save final model, scaler, and best threshold to pickle file.
    TODO:
    - Open model_path in write-binary mode
    - Use pickle.dump to store {"model": ..., "scaler": ..., "threshold": ...}
    """
    model_dir = {
        "model": model,
        "scaler": scaler,
        "threshold": threshold
    }
    with open(model_path, 'wb') as f:
        pickle.dump(model_dir, f ,protocol=pickle.HIGHEST_PROTOCOL)
def save_evaluation_results(metrics, results_path):
    """
    Save evaluation results (all metrics + params) to JSON file.
    TODO:
    - Use save_results_to_json() to save metrics to results_path
    """
    save_results_to_json(metrics, results_path)
    
def find_best_threshold_and_save(model, X_val, y_val, threshold_path):
    """
    Find the best threshold on validation set and save it.
    TODO:
    - Use find_best_threshold() to get the best threshold
    - Save the threshold to threshold_path
    """
    threshold, best_f1 = find_best_threshold(model, X_val, y_val)
    with open(threshold_path, 'w') as f:
        f.write(f"Best threshold: {threshold}\n")
        f.write(f"Best F1 score: {best_f1}\n")
    return threshold, best_f1


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
    X_train, y_train, X_val, y_val, scaler = load_and_preprocess(args.train_path, args.val_path, args.sampling)
    
    log_reg, log_reg_metrics = train_logistic_regression(X_train, y_train, X_val, y_val)
    
    rf, rf_metrics = train_random_forest(X_train, y_train, X_val, y_val, args.n_estimators, args.max_depth)
    
    mlp, mlp_metrics = train_MLPClassifier(X_train, y_train, X_val, y_val)
    
    voting_clf, voting_metrics = build_voting_classifier(log_reg, rf, mlp, X_train, y_train, X_val, y_val)
    
    threshold, best_f1 = find_best_threshold_and_save(voting_clf, X_val, y_val, args.threshold_path)
    
    save_model(voting_clf, scaler, threshold, args.model_path)
    
    save_evaluation_results(voting_metrics, args.results_path)

    pass


if __name__ == "__main__":
   
    parser = argparse.ArgumentParser(description="Credit Card Fraud Detection Training")
    
    
    parser.add_argument('--train_path', type=str, default='data/train.csv', help='Path to training data')
    parser.add_argument('--val_path', type=str, default='data/val.csv', help='Path to validation data')
    parser.add_argument('--model_path', type=str, default='models/fraud_model.pkl', help='Path to save the trained model')
    parser.add_argument('--results_path', type=str, default='results/evaluation_results.json', help='Path to save evaluation results')
    parser.add_argument('--threshold_path', type=str, default='results/best_threshold.txt', help='Path to save the best threshold')
    parser.add_argument('--sampling', type=str, default='none', help='Sampling strategy for training data (e.g., "none", "under", "over")')
    parser.add_argument('--sampling_method', type=str, default='random', help='Method for sampling (e.g., "random", "smote")')
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees in Random Forest')
    parser.add_argument('--max_depth', type=int, default=10, help='Maximum depth of trees in Random Forest')
    
    
    args = parser.parse_args()
    
    
    train_model(args)

    print("✅ Training completed successfully.")
    print(f"📦 Model saved to {args.model_path}")
    print(f"📊 Evaluation results saved to {args.results_path}")

   

