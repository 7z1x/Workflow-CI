import pandas as pd
import mlflow
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import json
import mpld3
import os
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE

X_train_scaled = pd.read_csv('breast_cancer_dataset_preprocessing/X_train_scaled.csv')
X_test_scaled = pd.read_csv('breast_cancer_dataset_preprocessing/X_test_scaled.csv')
y_train = pd.read_csv('breast_cancer_dataset_preprocessing/y_train.csv')
y_test = pd.read_csv('breast_cancer_dataset_preprocessing/y_test.csv')

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

mlflow.set_experiment("modelAutolog")
mlflow.xgboost.autolog()

with mlflow.start_run():
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train_smote, y_train_smote)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    print("Model dilatih dan dicatat di MLflow.")
    print(f"Accuracy: {acc}")
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    print(f"F1 Score: {f1}")

    metric_dict = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    }
    with open("metric_info.json", "w") as f:
        json.dump(metric_dict, f, indent=4)
    mlflow.log_artifact("metric_info.json")

    explainer = shap.Explainer(model, X_train_smote)
    shap_values = explainer(X_test_scaled)

    shap.plots.beeswarm(shap_values, show=False)
    plt.tight_layout()
    html_fig = mpld3.fig_to_html(plt.gcf())
    with open("estimator.html", "w") as f:
        f.write(html_fig)
    mlflow.log_artifact("estimator.html")
    plt.close()

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()
