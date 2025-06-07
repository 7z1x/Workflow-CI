import pandas as pd
import mlflow
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import json
import os
import seaborn as sns
from mlflow.models.signature import infer_signature # Untuk signature
from mlflow.types.schema import Schema, ColSpec # Untuk signature

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

# Jika X_train_smote bukan Pandas DataFrame, konversi dulu untuk input example
if not isinstance(X_train_smote, pd.DataFrame):
    # Asumsikan kolomnya sama dengan X_train_scaled atau perlu didefinisikan
    # Untuk contoh ini, kita buat nama kolom generik jika tidak ada
    try:
        input_column_names = X_train_scaled.columns.tolist()
    except AttributeError: # Jika X_train_scaled juga bukan DataFrame atau tidak punya columns
        input_column_names = [f"feature_{i}" for i in range(X_train_smote.shape[1])]
    
    X_train_smote_df = pd.DataFrame(X_train_smote, columns=input_column_names)
else:
    X_train_smote_df = X_train_smote


# Nonaktifkan autolog untuk model, kita akan log secara manual dengan signature
mlflow.xgboost.autolog(log_models=False, log_input_examples=False, log_model_signatures=False, log_datasets=False)

model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train_smote, y_train_smote) # Latih dengan data asli (NumPy array atau DataFrame)

y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] # Ambil probabilitas kelas positif

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

mlflow.log_metric("accuracy", acc)
mlflow.log_metric("precision", prec)
mlflow.log_metric("recall", rec)
mlflow.log_metric("f1_score", f1)

active_run_obj = mlflow.active_run()
if active_run_obj:
    run_id = active_run_obj.info.run_id
    print(f"MLflow Run ID (from active_run): {run_id}")
    with open("mlflow_run_id.txt", "w") as f:
        f.write(run_id)
    mlflow.log_artifact("mlflow_run_id.txt", "run_info")

    # Siapkan input example dari X_train_smote_df (harus Pandas DataFrame)
    # Kita hanya butuh satu sampel untuk input_example
    input_example = X_train_smote_df.head(1) 
    
    # Buat prediksi dummy dengan input_example untuk infer_signature
    # Beberapa model membutuhkan input dalam format tertentu untuk predict() saat infer_signature
    # Untuk XGBoost, biasanya DataFrame sudah cukup.
    try:
        predictions_for_signature = model.predict(input_example)
    except Exception as e:
        print(f"Warning: Could not generate predictions for signature inference: {e}")
        # Jika predict gagal dengan DataFrame, coba konversi ke NumPy array
        try:
            predictions_for_signature = model.predict(input_example.to_numpy())
        except Exception as e_np:
            print(f"Warning: Could not generate predictions for signature (NumPy fallback) inference: {e_np}")
            predictions_for_signature = None # Tidak bisa infer output signature

    # Definisikan signature
    if predictions_for_signature is not None:
        # MLServer mengharapkan input dengan nama tertentu jika menggunakan V2 protocol
        # Kita akan sesuaikan input schema agar cocok dengan "input-0" yang kita gunakan di sample.json
        # Buat input schema secara manual agar nama inputnya "input-0"
        # dan tipenya float (karena datanya sudah di-scale) untuk semua 30 fitur.
        input_schema = Schema([ColSpec(type="double", name="input-0")] * X_train_smote_df.shape[1]) # Buat 30 kolom input "input-0"
                                                                                                 # Atau, jika ingin nama kolom asli:
                                                                                                 # input_schema = Schema([ColSpec(type="double", name=col) for col in input_example.columns])
        output_schema = Schema([ColSpec(type="integer")]) # Asumsi output adalah integer (0 atau 1)
        signature = infer_signature(input_example, predictions_for_signature) # Ini akan mencoba infer
        
        # Ganti nama input di signature yang di-infer agar sesuai dengan apa yang kita kirim ("input-0")
        # Ini adalah bagian yang agak tricky, MLServer V2 protocol lebih suka tensor dengan nama generik.
        # Cara lebih mudah: kita definisikan schema input agar MLServer bisa menerjemahkan.
        # Namun, untuk XGBoost yang dilatih dengan DataFrame, signature yang di-infer dengan nama kolom asli biasanya lebih baik.
        # Mari kita coba infer signature apa adanya dulu, dan sesuaikan sample.json jika perlu.
        # Atau kita bisa definisikan input tensor secara eksplisit.
        # Untuk sekarang, kita log dengan signature yang di-infer dari DataFrame asli.
        
        # Untuk MLServer V2 dan input bernama "input-0", kita bisa coba buat signature seperti ini:
        # Input: satu tensor bernama "input-0" dengan shape (n_samples, n_features)
        # Output: satu tensor (misalnya, prediksi kelas)
        # Ini lebih cocok jika modelmu dibungkus dengan cara tertentu.
        # Untuk MLflow pyfunc, signature yang di-infer dari DataFrame biasanya lebih baik.

        # Kita akan gunakan signature yang di-infer dari input_example (DataFrame) dan outputnya.
        # Ini akan membuat model mengharapkan input DataFrame dengan nama kolom yang sama dengan input_example.
        
        print(f"Signature to be logged: {signature}")

    else:
        signature = None
        print("Warning: Could not infer signature due to prediction failure on input_example.")


    print(f"Manually logging XGBoost model to run ID: {run_id}")
    mlflow.xgboost.log_model(
        xgb_model=model,
        artifact_path="model",
        signature=signature, # Tambahkan signature
        input_example=input_example # Tambahkan input_example
    )
    print("XGBoost model manually logged with signature and input example.")

else:
    print("Error: No active MLflow run found by mlflow.active_run(). Cannot save run_id or log model.")
    exit(1) 

# ... (sisa kode untuk logging metrik, artefak plot, dll. tetap sama) ...

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
    explainer = shap.TreeExplainer(model, X_train_smote_df) # Gunakan DataFrame jika model dilatih dengan itu atau explainer mengharapkannya
    shap_values = explainer.shap_values(X_test_scaled) 
    
    plt.figure()
    shap.summary_plot(shap_values, X_test_scaled, feature_names=X_train_smote_df.columns, show=False, plot_size=(8,5)) # Tambahkan feature_names
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
