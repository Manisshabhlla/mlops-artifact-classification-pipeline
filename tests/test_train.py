import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from src.train import train_model

def test_config_loading():
    print("Running test_config_loading...")
    with open("config/config.json", "r") as f:
        config = json.load(f)
    assert isinstance(config["C"], float)
    assert isinstance(config["solver"], str)
    assert isinstance(config["max_iter"], int)
    print("test_config_loading passed.")

def test_model_training():
    print("Running test_model_training...")
    digits = load_digits()
    X, y = digits.data, digits.target
    with open("config/config.json", "r") as f:
        config = json.load(f)
    model = train_model(X, y, config)
    assert isinstance(model, LogisticRegression)
    assert hasattr(model, "coef_")
    print("test_model_training passed.")

def test_model_accuracy():
    print("Running test_model_accuracy...")
    digits = load_digits()
    X, y = digits.data, digits.target
    with open("config/config.json", "r") as f:
        config = json.load(f)
    model = train_model(X, y, config)
    acc = model.score(X, y)
    print("Accuracy:", acc)
    assert acc > 0.85
    print("test_model_accuracy passed.")
