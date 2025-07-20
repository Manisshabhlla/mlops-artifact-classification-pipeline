import json
import joblib
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression

def load_config(path="config/config.json"):
    print(f"Loading config from {path}...")
    with open(path, "r") as f:
        config = json.load(f)
    print("Config loaded:", config)
    return config

def train_model(X, y, config):
    print("Training Logistic Regression model...")
    model = LogisticRegression(
        C=config["C"],
        solver=config["solver"],
        max_iter=config["max_iter"]
    )
    model.fit(X, y)
    print("Model training complete.")
    return model

if __name__ == "__main__":
    print("Starting training process...")
    config = load_config()
    digits = load_digits()
    X, y = digits.data, digits.target
    print(f"Loaded dataset: {X.shape[0]} samples with {X.shape[1]} features.")
    model = train_model(X, y, config)
    joblib.dump(model, "model_train.pkl")
    print("Model saved as model_train.pkl")
