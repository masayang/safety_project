from datasets import load_dataset
import pandas as pd
import random
from config import Config


def create_user_interactions():
    """Create user interactions dataset from SQuAD dataset."""
    # 1. Load dataset
    dataset = load_dataset("squad", split="train[:1000]")

    # 2. Create dataframe
    data = []
    for item in dataset:
        q = item["question"]
        a = item["answers"]["text"][0]
        # Random assignment of 80% truthful, 20% deceptive
        label = "truthful" if random.random() > 0.2 else "deceptive"
        # For deceptive cases, generate a simple incorrect answer (simplified simulation)
        answer = a if label == "truthful" else f"Wrong answer: {a} is incorrect."
        data.append({"question": q, "answer": answer, "label": label})

    df = pd.DataFrame(data)

    # 3. Check distribution
    truthful_count = len(df[df["label"] == "truthful"])
    deceptive_count = len(df[df["label"] == "deceptive"])
    print(f"Truthful: {truthful_count}, Deceptive: {deceptive_count}")

    # 4. Save
    df.to_csv(Config.USER_INTERACTIONS_PATH, index=False)
    print(f"Updated dataset saved to {Config.USER_INTERACTIONS_PATH}")


if __name__ == '__main__':
    create_user_interactions()