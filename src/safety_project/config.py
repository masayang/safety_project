import os


class Config:
    DATA_DIR = "data"
    OUTPUT_DIR = "images"
    
    DATA_PATH = os.path.join(DATA_DIR, "eval_results.csv")
    METRICS_BAR_PATH = os.path.join(OUTPUT_DIR, "metrics_bar.png")
    LABEL_DISTRIBUTION_PIE_PATH = os.path.join(OUTPUT_DIR, "label_distribution_pie.png")
    QUESTION_PREDICTIONS_BAR_PATH = os.path.join(OUTPUT_DIR, "question_predictions_bar.png")
    USER_INTERACTIONS_PATH = os.path.join(DATA_DIR, "user_interactions.csv")

    MAX_CORE_WORKERS = 4
    BATCH_SIZE = 16