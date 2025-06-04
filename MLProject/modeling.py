import pandas as pd
import mlflow
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import json
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

mlflow.set_experiment("Breast_Cancer_CI_Experiment") 
mlflow.xgboost.autolog(log_models=True, log_input_examples=False, log_model_signatures=False) 

with mlflow.start_run() as run:
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        use_label_encoder=False, 
        eval_metric='logloss',
        random_state=42 
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

    run_id = run.info.run_id
    print(f"MLflow Run ID: {run_id}")
    with open("mlflow_run_id.txt", "w") as f:
        f.write(run_id)
    mlflow.log_artifact("mlflow_run_id.txt", "run_info")
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
    mlflow.log_artifact("metric_info.json", "evaluation_reports")

    try:
        print("📊 Menghitung SHAP values...")
        explainer = shap.TreeExplainer(model, X_train_smote)
        shap_values = explainer.shap_values(X_test_scaled)
        
        plt.figure() 
        shap.summary_plot(shap_values, X_test_scaled, show=False, plot_size=(8,5))
        shap_img_path = "shap_summary.png"
        plt.savefig(shap_img_path, bbox_inches="tight")
        mlflow.log_artifact(shap_img_path, "shap_plots")
        plt.close()
    except Exception as e:
        print(f"⚠️ Gagal membuat SHAP summary plot: {e}")

    try:
        plt.figure(figsize=(6, 5))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path, "evaluation_plots") 
        plt.close()
    except Exception as e:
        print(f"⚠️ Gagal membuat confusion matrix plot: {e}")

print("\n🎉 Selesai! MLflow berjalan lokal, cek folder 'mlruns'.")
