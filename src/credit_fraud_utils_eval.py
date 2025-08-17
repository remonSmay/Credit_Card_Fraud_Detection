import json
from sklearn.metrics import f1_score, average_precision_score
import numpy as np

def evaluate_model(model, X_val, y_val):
    """
    Evaluate model using F1-score and PR-AUC.
    Args:
        model: Trained model with predict and predict_proba methods.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation labels.
    Returns:
        dict: Dictionary containing F1-score and PR-AUC
    """
    # how get predictions and probabilities
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1] # get probabilities for the positive class (fraud)--> 1
    
    f1 = f1_score(y_val, y_pred)
    pr_auc = average_precision_score(y_val, y_prob)
    
    return {"f1_score": f1, "pr_auc": pr_auc}
    
    


def find_best_threshold(model, X_val, y_val):
    """
    Find threshold that maximizes F1-score.
    Args:
        model: Trained model with predict_proba method.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation labels.
    Returns:
        float: Best threshold for F1-score.
    """
    y_prob = model.predict_proba(X_val)[:, 1] 
    best_f1 = 0
    best_threshold = 0.5

    for threshold in np.arange(0.0, 1.0, 0.01):
        y_pred_threshold = (y_prob >= threshold).astype(int)
        current_f1 = f1_score(y_val, y_pred_threshold)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold
            
    return best_threshold , best_f1



def save_results_to_json(results, json_path):
    """
    Save results dictionary to a JSON file.

    Args:
        results (dict): A dictionary containing the results to save.
        json_path (str): The path to the output JSON file.
    """
    try:
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Results successfully saved to {json_path}")
    except IOError as e:
        print(f"Error saving results to {json_path}: {e}")
