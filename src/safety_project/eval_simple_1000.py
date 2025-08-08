import pandas as pd
from transformers import pipeline
from sklearn.metrics import accuracy_score, f1_score
from concurrent.futures import ThreadPoolExecutor
from config import Config

def process_answer(answer, classifier):
    return classifier(answer)[0]["label"]

def main():
    """Evaluate model on user interactions dataset."""
    # 1. Load data
    df = pd.read_csv(Config.USER_INTERACTIONS_PATH)

    # 2. Load model
    classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

    # 3. Inference with thread-based parallel processing
    with ThreadPoolExecutor(max_workers=Config.MAX_CORE_WORKERS) as executor:
        results = list(executor.map(lambda x: process_answer(x, classifier), df["answer"].tolist()))
    df["predicted"] = results
    df["predicted_label"] = df["predicted"].apply(lambda x: "truthful" if x == "POSITIVE" else "deceptive")

    # 4. Evaluation
    accuracy = accuracy_score(df["label"], df["predicted_label"])
    f1 = f1_score(df["label"], df["predicted_label"], pos_label="truthful")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"F1 Score: {f1:.2f}")

    # 5. Save results
    df.to_csv(Config.DATA_PATH)
    print(f"Evaluation results saved to {Config.DATA_PATH}")

if __name__ == '__main__':
    main()
