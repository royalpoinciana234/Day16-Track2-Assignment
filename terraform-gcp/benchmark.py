import time
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def run_benchmark():
    print("=== STARTING LIGHTGBM BENCHMARK ===")
    
    # 1. Load Data
    start_load = time.time()
    try:
        df = pd.read_csv('creditcard.csv')
    except FileNotFoundError:
        print("creditcard.csv not found in the current directory.")
        print("Please make sure the dataset is downloaded and extracted in this folder.")
        return
    
    load_time = time.time() - start_load
    print(f"Data load time: {load_time:.4f} seconds")

    # 2. Preprocessing
    X = df.drop(columns=['Class'])
    y = df['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Training
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.1,
        'num_leaves': 31,
        'max_depth': -1,
        'feature_fraction': 0.8,
        'verbose': -1,
        'n_jobs': -1 # Use all vCPUs
    }
    
    print("Training LightGBM model...")
    start_train = time.time()
    
    # Using lightgbm train API
    evals_result = {}
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[val_data],
        callbacks=[lgb.record_evaluation(evals_result)]
    )
    
    train_time = time.time() - start_train
    print(f"Training time: {train_time:.4f} seconds")
    
    # Get best iteration
    best_iteration = model.best_iteration if hasattr(model, 'best_iteration') else 100
    if best_iteration == 0:
        best_iteration = 100
    
    # 4. Evaluation
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob >= 0.5).astype(int)
    
    auc_roc = roc_auc_score(y_test, y_pred_prob)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Best iteration: {best_iteration}")
    print(f"AUC-ROC: {auc_roc:.6f}")
    print(f"Accuracy: {accuracy:.6f}")
    print(f"F1-Score: {f1:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    
    # 5. Inference Latency (1 row)
    single_row = X_test.iloc[[0]]
    latencies = []
    # Warmup
    for _ in range(10):
        _ = model.predict(single_row)
        
    for _ in range(100):
        t0 = time.time()
        _ = model.predict(single_row)
        latencies.append(time.time() - t0)
        
    avg_latency = np.mean(latencies) * 1000  # in ms
    print(f"Inference latency (1 row): {avg_latency:.4f} ms")
    
    # 6. Inference Throughput (1000 rows)
    subset_1000 = X_test.iloc[:1000]
    # Warmup
    for _ in range(5):
        _ = model.predict(subset_1000)
        
    t0 = time.time()
    for _ in range(50):
        _ = model.predict(subset_1000)
    total_time = time.time() - t0
    
    throughput = (1000 * 50) / total_time  # rows per second
    print(f"Inference throughput (1000 rows): {throughput:.2f} rows/sec")
    
    # Save results to JSON
    results = {
        "data_load_time_sec": round(load_time, 4),
        "training_time_sec": round(train_time, 4),
        "best_iteration": int(best_iteration),
        "auc_roc": round(auc_roc, 6),
        "accuracy": round(accuracy, 6),
        "f1_score": round(f1, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "inference_latency_1row_ms": round(avg_latency, 4),
        "inference_throughput_1000rows_sec": round(throughput, 2)
    }
    
    with open('benchmark_result.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("Results saved to benchmark_result.json")

if __name__ == "__main__":
    run_benchmark()
