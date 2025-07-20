import joblib
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report, accuracy_score
from tabulate import tabulate

def inference():
    print("🔍 Loading trained model...")
    model = joblib.load("model_train.pkl")
    print("✅ Model loaded.")

    digits = load_digits()
    X = digits.data
    y = digits.target
    print(f"📊 Running inference on {X.shape[0]} samples...")

    predictions = model.predict(X)

    # Display first 10 predictions
    results = [[i, y[i], predictions[i]] for i in range(10)]
    print("\n📋 Inference Results (First 10 Samples):")
    print(tabulate(results, headers=["Index", "Actual", "Predicted"], tablefmt="grid"))

    # Classification report
    report_dict = classification_report(y, predictions, output_dict=True)
    report_table = []

    for label, metrics in report_dict.items():
        if isinstance(metrics, dict):
            report_table.append([
                label,
                f"{metrics['precision']:.4f}",
                f"{metrics['recall']:.4f}",
                f"{metrics['f1-score']:.4f}",
                int(metrics['support'])
            ])
        else:
            # Handle accuracy separately
            report_table.append([
                label,
                "",
                "",
                f"{metrics:.4f}",
                ""
            ])

    print("\n📈 Classification Report:")
    print(tabulate(report_table, headers=["Label", "Precision", "Recall", "F1-Score", "Support"], tablefmt="grid"))

    acc = accuracy_score(y, predictions)
    print(f"\n✅ Overall Accuracy: {acc:.4f}")

if __name__ == "__main__":
    print("🚀 Starting inference...")
    inference()
