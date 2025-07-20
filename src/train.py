import json
import joblib
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression

def load_config(path="config/config.json"):
    with open(path, "r") as f:
        return json.load(f)

def train_model(X, y, config):
    model = LogisticRegression(
        C=config["C"],
        solver=config["solver"],
        max_iter=config["max_iter"]
    )
    model.fit(X, y)
    return model

if __name__ == "__main__":
    config = load_config()
    digits = load_digits()
    X, y = digits.data, digits.target
    model = train_model(X, y, config)
    joblib.dump(model, "model_train.pkl")
