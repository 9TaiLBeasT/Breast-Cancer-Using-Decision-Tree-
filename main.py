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

    X = data.drop(["diagnosis"], axis=1)
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
    print("Accuracy (holdout): ", accuracy_score(y_test, y_pred))
    print("Confusion matrix (holdout):\n", confusion_matrix(y_test, y_pred))
    print("Classification report (holdout):\n", classification_report(y_test, y_pred))

    # ROC-AUC requires probabilities
    y_proba = model.predict_proba(X_test)[:, 1]
    print("ROC-AUC (holdout): ", roc_auc_score(y_test, y_proba))

    # Cross-validated predictions (more stable estimate than a single split)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    print("ROC-AUC (5-fold CV): ", roc_auc_score(y, cv_proba))

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
