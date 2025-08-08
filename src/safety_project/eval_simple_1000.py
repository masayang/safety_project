import pandas as pd
from transformers import pipeline
from sklearn.metrics import accuracy_score, f1_score
from concurrent.futures import ThreadPoolExecutor
from config import Config

def process_answer(answer, classifier):
    return classifier(answer)[0]["label"]

def process_batch(answers, classifier):
    return [pred["label"] for pred in classifier(answers, batch_size=Config.BATCH_SIZE)]

def main():
    """Evaluate model on user interactions dataset."""
    # 1. Load data
    df = pd.read_csv(Config.USER_INTERACTIONS_PATH)

    # 2. Load model
    classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

    # データをバッチに分割
    batch_size = 64
    answer_batches = [df["answer"].tolist()[i:i+batch_size] for i in range(0, len(df), batch_size)]
    
    # 3. Inference with thread-based parallel processing
    with ThreadPoolExecutor(max_workers=Config.MAX_CORE_WORKERS) as executor:
        results = []
        for batch in answer_batches:
            results.extend(executor.submit(process_batch, batch, classifier).result())
    
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
