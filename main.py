import pandas as pd
import pickle

from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier, export_text


def create_model(data):
    """Train a Decision Tree classifier using Information Gain (entropy).

    Notes:
    - scikit-learn's DecisionTreeClassifier implements CART-style trees.
    - Using criterion='entropy' uses information gain for splits (close to the J48/C4.5 spirit,
      though it is not a full C4.5 implementation with gain ratio + pruning).
    """

    # Canonical feature order matching app.py
    feature_cols = [
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
        "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
        "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
        "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se",
        "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
        "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
    ]
    
    # Ensure data has all these columns
    missing_cols = set(feature_cols) - set(data.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")

    X = data[feature_cols]
    y = data["diagnosis"]

    # Holdout split for a quick evaluation report
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = DecisionTreeClassifier(
        criterion="entropy",  # information gain
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    
    print("Accuracy (holdout): ", acc)
    print("Confusion matrix (holdout):\n", cm)
    print("Classification report (holdout):\n", cr)

    # ROC-AUC requires probabilities
    y_proba = model.predict_proba(X_test)[:, 1]
    roc_holdout = roc_auc_score(y_test, y_proba)
    print("ROC-AUC (holdout): ", roc_holdout)

    # Cross-validated predictions (more stable estimate than a single split)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    roc_cv = roc_auc_score(y, cv_proba)
    print("ROC-AUC (5-fold CV): ", roc_cv)
    
    # Save metrics to file
    with open("model_metrics.txt", "w") as f:
        f.write(f"Accuracy (holdout): {acc}\n")
        f.write(f"Confusion matrix (holdout):\n{cm}\n")
        f.write(f"Classification report (holdout):\n{cr}\n")
        f.write(f"ROC-AUC (holdout): {roc_holdout}\n")
        f.write(f"ROC-AUC (5-fold CV): {roc_cv}\n")

    # Print a readable rule representation (top of the tree). Useful for explainability.
    print("\nDecision tree rules (text):")
    print(export_text(model, feature_names=list(X.columns)))

    return model


def get_clean_data():
    data = pd.read_csv("Dataset/merged_data.csv")
    data = data.drop(["Unnamed: 32", "id"], axis=1, errors='ignore')

    data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})
    return data


def main():
    data = get_clean_data()
    model = create_model(data)

    # Save the model (inference expects this file)
    with open("model.pkl", "wb") as file:
        pickle.dump(model, file)


if __name__ == "__main__":
    main()
