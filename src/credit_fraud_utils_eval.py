import json

def evaluate_model(model, X_val, y_val):
    """
    Evaluate model using F1-score and PR-AUC.
    TODO:
    - Get predictions & probabilities
    - Calculate metrics
    - Return scores dictionary { "f1": ..., "pr_auc": ... }
    """
    pass


def find_best_threshold(model, X_val, y_val):
    """
    Find threshold that maximizes F1-score.
    TODO:
    - Loop over possible thresholds
    - Evaluate F1-score for each
    - Return best threshold value
    """
    pass


def save_results_to_json(results, json_path):
    """
    Save results dictionary to a JSON file.
    TODO:
    - Open file at json_path
    - Use json.dump to save results with indent=4
    Example results dict:
    {
        "model_name": "RandomForest",
        "f1_score": 0.92,
        "pr_auc": 0.88,
        "best_threshold": 0.45,
        "parameters": { "n_estimators": 50, "max_depth": 8 }
    }
    """
    pass
