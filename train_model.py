import pandas as pd
import numpy as np
import re
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 60)
print("NEWS CATEGORY CLASSIFICATION - ML PROJECT")
print("=" * 60)

print("\n[1] Loading dataset...")

df = pd.read_csv("data/News.csv")

print("Dataset loaded successfully!")
print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 2. BASIC DATASET INFORMATION
# ============================================================

print("\n[2] Dataset information")

print("\nMissing values:")
print(df.isnull().sum())

print("\nCategory distribution:")
print(df["Category"].value_counts())


# ============================================================
# 3. REMOVE MISSING VALUES
# ============================================================

print("\n[3] Handling missing values...")

df = df.dropna(subset=["Text", "Category"])

print("Rows after removing missing values:", len(df))


# ============================================================
# 4. REMOVE DUPLICATES
# ============================================================

print("\n[4] Removing duplicate articles...")

before_duplicates = len(df)

df = df.drop_duplicates(subset=["Text"])

after_duplicates = len(df)

print("Duplicates removed:", before_duplicates - after_duplicates)
print("Rows remaining:", after_duplicates)


# ============================================================
# 5. TEXT PREPROCESSING
# ============================================================

print("\n[5] Cleaning text...")


def clean_text(text):
    """
    Basic text preprocessing:
    - Convert text to lowercase
    - Remove URLs
    - Remove HTML tags
    - Remove special characters
    - Remove extra spaces
    """

    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Keep only letters and numbers
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


df["clean_text"] = df["Text"].apply(clean_text)

print("Text preprocessing completed.")


# ============================================================
# 6. PREPARE FEATURES AND TARGET
# ============================================================

X = df["clean_text"]
y = df["Category"]


# ============================================================
# 7. TRAIN-TEST SPLIT
# ============================================================

print("\n[6] Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 8. TF-IDF FEATURE EXTRACTION
# ============================================================

print("\n[7] Applying TF-IDF...")

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF completed.")
print("Training feature shape:", X_train_tfidf.shape)
print("Testing feature shape:", X_test_tfidf.shape)


# ============================================================
# 9. DEFINE ML MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
}


# ============================================================
# 10. TRAIN AND EVALUATE MODELS
# ============================================================

results = {}

print("\n")
print("=" * 60)
print("MODEL TRAINING AND EVALUATION")
print("=" * 60)


for model_name, model in models.items():

    print("\n" + "-" * 60)
    print("Training:", model_name)
    print("-" * 60)

    # Train
    model.fit(X_train_tfidf, y_train)

    # Predict
    y_pred = model.predict(X_test_tfidf)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # Store results
    results[model_name] = {
        "model": model,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predictions": y_pred
    }

    # Print metrics
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )


# ============================================================
# 11. MODEL COMPARISON
# ============================================================

print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

comparison = pd.DataFrame({
    "Model": list(results.keys()),
    "Accuracy": [
        results[name]["accuracy"]
        for name in results
    ],
    "Precision": [
        results[name]["precision"]
        for name in results
    ],
    "Recall": [
        results[name]["recall"]
        for name in results
    ],
    "F1-Score": [
        results[name]["f1"]
        for name in results
    ]
})

print("\n")
print(comparison.to_string(index=False))


# ============================================================
# 12. SELECT BEST MODEL
# ============================================================

best_model_name = max(
    results,
    key=lambda name: results[name]["f1"]
)

best_model = results[best_model_name]["model"]

print("\n")
print("=" * 60)
print("BEST MODEL")
print("=" * 60)

print("Best Model:", best_model_name)
print(
    "Best F1-Score:",
    f"{results[best_model_name]['f1']:.4f}"
)


# ============================================================
# 13. CONFUSION MATRICES
# ============================================================

print("\nGenerating confusion matrices...")

for model_name in results:

    y_pred = results[model_name]["predictions"]

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=sorted(y.unique())
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=sorted(y.unique())
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    disp.plot(
        ax=ax,
        xticks_rotation=45
    )

    plt.title(
        f"Confusion Matrix - {model_name}"
    )

    plt.tight_layout()

    filename = (
        model_name.lower()
        .replace(" ", "_")
        + "_confusion_matrix.png"
    )

    plt.savefig(filename, dpi=300)

    plt.show()

    plt.close()


# ============================================================
# 14. SAVE BEST MODEL
# ============================================================

print("\nSaving best model...")

with open("model.pkl", "wb") as file:
    pickle.dump(best_model, file)

print("Saved: model.pkl")


# ============================================================
# 15. SAVE TF-IDF VECTORIZER
# ============================================================

with open("vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("Saved: vectorizer.pkl")


# ============================================================
# 16. SAVE MODEL COMPARISON
# ============================================================

comparison.to_csv(
    "model_comparison.csv",
    index=False
)

print("Saved: model_comparison.csv")


# ============================================================
# 17. FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nGenerated files:")

print("1. model.pkl")
print("2. vectorizer.pkl")
print("3. model_comparison.csv")
print("4. logistic_regression_confusion_matrix.png")
print("5. decision_tree_confusion_matrix.png")
print("6. random_forest_confusion_matrix.png")

print("\nBest Model:", best_model_name)

print("\nYou can now use the trained model with Streamlit.")