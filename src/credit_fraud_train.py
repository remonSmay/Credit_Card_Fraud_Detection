import json
import os
import pickle
import warnings
from types import SimpleNamespace

from credit_fraud_utils_data import load_data, preprocess_data
from credit_fraud_utils_eval import evaluate_model, find_best_threshold, save_results_to_json

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")


class CreditFraudPipeline:
    def __init__(self, config):
        self.args = SimpleNamespace(**config)
        self.experiment_number = 1
        os.makedirs(self.args.results_dir, exist_ok=True)

    def load_and_preprocess(self):
        df_train = load_data(self.args.train_path)
        df_val = load_data(self.args.val_path)

        # preprocess with sampling for train
        X_train, y_train = preprocess_data(df_train, sampling_method=self.args.sampling, sampling_strategy=0.5)

        # preprocess val (no sampling)
        X_val, y_val = preprocess_data(df_val)

        return X_train, y_train, X_val, y_val

    def train_models(self, X_train, y_train, hyperparams):
        # Logistic Regression
        log_reg = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=42)
        log_reg.fit(X_train, y_train)

        # Random Forest (using hyperparameters)
        rf = RandomForestClassifier(
            n_estimators=hyperparams.get("n_estimators", 100),
            max_depth=hyperparams.get("max_depth", None),
            class_weight="balanced",
            random_state=42
        )
        rf.fit(X_train, y_train)

        # MLP Classifier
        mlp = MLPClassifier(
            hidden_layer_sizes=(16, 32),
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            solver="adam",
            validation_fraction=0.1
        )
        mlp.fit(X_train, y_train)

        # Voting Classifier
        voting_clf = VotingClassifier(
            estimators=[("log_reg", log_reg), ("rf", rf), ("mlp", mlp)],
            voting="soft"
        )
        voting_clf.fit(X_train, y_train)

        return voting_clf

    def evaluate_and_save(self, model, X_val, y_val, hyperparams):
        metrics = evaluate_model(model, X_val, y_val)

        # threshold tuning
        threshold, best_f1 = find_best_threshold(model, X_val, y_val)
        metrics["best_threshold"] = threshold
        metrics["best_f1"] = best_f1
        metrics["hyperparameters"] = hyperparams

        # save results JSON
        json_file_name = f"experiment_{self.experiment_number}.json"
        json_path = os.path.join(self.args.results_dir, json_file_name)
        save_results_to_json(metrics, json_path)

        # save model + threshold
        model_file_name = f"fraud_model_exp_{self.experiment_number}.pkl"
        model_path = os.path.join("models", model_file_name)
        os.makedirs("models", exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump({"model": model, "threshold": threshold}, f)

        print(f"✅ Experiment {self.experiment_number} saved: {json_path}, {model_path}")
        self.experiment_number += 1

    def run(self):
        X_train, y_train, X_val, y_val = self.load_and_preprocess()

        for hyperparams in self.args.hyperparameters:
            model = self.train_models(X_train, y_train, hyperparams)
            self.evaluate_and_save(model, X_val, y_val, hyperparams)


# ========================
# Main Entry
# ========================
if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)

    pipeline = CreditFraudPipeline(config)
    pipeline.run()
