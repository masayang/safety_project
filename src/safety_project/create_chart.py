import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from config import Config


def categorize_question(q):
    """Categorize questions into simple categories (e.g., keyword-based)."""
    q = q.lower()
    if "notre dame" in q:
        return "Schools"
    elif "beyonce" in q:
        return "Celebrities"
    elif "ronald reagan" in q:
        return "Celebrities"
    elif "song" in q:
        return "Songs"
    else:
        return "Other"


def create_metrics_bar_chart(df):
    """Create a bar chart for Accuracy and F1 Score."""
    # Calculate metrics
    accuracy = accuracy_score(df["label"], df["predicted_label"])
    f1 = f1_score(df["label"], df["predicted_label"], pos_label="truthful")

    # Bar chart (Accuracy and F1 Score)
    plt.figure(figsize=(8, 5))
    plt.bar(["Accuracy", "F1 Score"], [accuracy, f1], color=["blue", "green"])
    plt.title("Model Evaluation Metrics")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    for i, v in enumerate([accuracy, f1]):
        plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
    plt.savefig(Config.METRICS_BAR_PATH)
    plt.close()
    
    return accuracy, f1


def create_label_distribution_pie_chart(df):
    """Create a pie chart for the distribution of Truthful/Deceptive labels."""
    # Pie chart (Distribution of Truthful/Deceptive)
    label_counts = df["label"].value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(
        label_counts,
        labels=label_counts.index,
        autopct="%1.1f%%",
        colors=["lightblue", "lightcoral"]
    )
    plt.title("Distribution of Truthful vs Deceptive Labels")
    plt.savefig(Config.LABEL_DISTRIBUTION_PIE_PATH)
    plt.close()


def create_question_predictions_bar_chart(df):
    """Create a bar chart for prediction results by question category."""
    # Categorize questions and create bar chart
    df["category"] = df["question"].apply(categorize_question)
    category_counts = df.groupby("category")["predicted_label"].value_counts().unstack().fillna(0)

    # Bar chart
    category_counts.plot(kind="bar", stacked=True, figsize=(10, 6), color=["#4682B4", "#FFA07A"])
    plt.title("Predicted Labels by Question Category")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.legend(title="Predicted Label", labels=["deceptive", "truthful"])
    plt.tight_layout()
    plt.savefig(Config.QUESTION_PREDICTIONS_BAR_PATH)
    plt.close()


def create_all_charts():
    """Create all visualization charts from the evaluation data."""
    # 1. Load data
    df = pd.read_csv(Config.DATA_PATH)
    
    print(df)
    
    # 2. Create charts
    accuracy, f1 = create_metrics_bar_chart(df)
    create_label_distribution_pie_chart(df)
    create_question_predictions_bar_chart(df)
    
    # 3. Result confirmation
    print(f"Visualizations saved: {Config.METRICS_BAR_PATH}, {Config.LABEL_DISTRIBUTION_PIE_PATH}, {Config.QUESTION_PREDICTIONS_BAR_PATH}")
    
    return df, accuracy, f1


if __name__ == '__main__':
    create_all_charts()