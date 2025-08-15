import datetime
import os

import joblib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GroupKFold, GridSearchCV, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import seaborn as sns

import pandas as pd

csvfile = "./Data/WESAD_filtered.csv"
model_name = "knn_WESAD_6F"
feature_names = ["MeanSCR", "MaxSCR", "MinSCR", "RangeSCR", "SkewenessSCR", "KurtosisSCR"]

def get_train_test_data(csv_file, feature_cols):
    df = pd.read_csv(csv_file, header=0)

    X = df[feature_cols].values
    y = df["label"].values
    groups = df["subject"].values

    # Train/test split creation
    group_shuffle = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=5) # We use group shuffle split so that same subject won't appear in both train and test set
    train_index, test_index = next(group_shuffle.split(X, y, groups))
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    groups_train = groups[train_index]
    groups_test = groups[test_index]

    return X_train, y_train, groups_train, X_test, y_test, groups_test, y


def get_data(csv_file, feature_cols):
    df = pd.read_csv(csv_file, header=0)

    X = df[feature_cols].values
    y = df["label"].values
    groups = df["subject"].values

    return X, y, groups

def train_knn_model(k_folds=10):

    pipeline = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier())])

    X_train, y_train, groups_train, X_test, y_test, groups_test, y = get_train_test_data(csvfile, feature_names)

    param_grid = {
        "knn__n_neighbors": [3, 5, 7, 9, 11],
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["euclidean", "manhattan", "minkowski"]
    }

    cross_valid = GroupKFold(k_folds)
    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro"
    }
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cross_valid, scoring=scoring, refit="f1_macro", n_jobs=-1)

    grid_search.fit(X_train, y_train, groups=groups_train)

    print("Best parameters:", grid_search.best_params_)

    best_model = grid_search.best_estimator_

    test_score(best_model, X_test, y_test, y, "file.txt")
    return best_model

def test_score(model, X_test, y_test, y, file=None):
    print("\nHold-out Test Set Results:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # F1 score
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_micro = f1_score(y_test, y_pred, average="micro")
    f1_weighted = f1_score(y_test, y_pred, average="macro")

    # Precision
    precision = precision_score(y_test, y_pred, average="macro")
    # Recall
    recall = recall_score(y_test, y_pred, average="macro")

    if file:
        with open(file, 'w') as f:
            lines = []
            lines.append(f"Accuracy: {accuracy}\n")
            lines.append(f"F1 Macro: {f1_macro}\n")
            lines.append(f"F1 Micro: {f1_micro}\n")
            lines.append(f"F1 Weighted: {f1_weighted}\n")
            lines.append(f"Precision: {precision}\n")
            lines.append(f"Recall: {recall}\n")
            f.writelines(lines)

    print_cm(y_test, y_pred, y)


def print_cm(y_test, y_pred, y):
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    fig.savefig("confusion_matrix.png")

def save_model(model):

    # Main model infos
    model_artifact = {
        "model": model,
        "feature_names": feature_names,
        "created_at": datetime.datetime.utcnow().isoformat(),
    }

    # Artifact saved as joblib file
    joblib.dump(model_artifact, f"{model_name}.joblib")


def load_model(joblib_file):

    artifact = joblib.load(joblib_file)
    model = artifact["model"]
    feature_names = artifact["feature_names"]

    return model, feature_names



def train_knn_main():
    model = train_knn_model()

    save_model(model)


#train_knn_main()

