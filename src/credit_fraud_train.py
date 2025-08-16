import argparse
from credit_fraud_utils_data import load_data, preprocess_data
from credit_fraud_utils_eval import evaluate_model, find_best_threshold, save_results_to_json
import pickle

def train_model(args):
    """
    Main training function.
    TODO:
    1. Load data (train, val) using load_data()
    2. Preprocess data (scaling, sampling) [y] 
    3. Train Logistic Regression and Random Forest
    4. Evaluate models → get metrics
    5. Select best models for Voting Classifier
    6. Find best threshold
    7. Save model + threshold to pickle
    8. Save results (metrics + params) to JSON
    """
    df= load_data(args.data_path)
    X , y =preprocess_data(df,args.sampling_strategy, args.sampling_method)
     
    pass 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credit Card Fraud Detection Training")
    # TODO: Add CLI arguments (data path, model type, save paths, etc.)
    args = parser.parse_args()
    train_model(args)
