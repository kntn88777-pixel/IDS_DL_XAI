# 02_train_models.py — PHIÊN BẢN CHẠY NGON 100% (đã test với IoT-23)
import joblib
import numpy as np                          # ← THÊM DÒNG NÀY!
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import xgboost as xgb
from config import Config
import os

# Tự động lấy tên dataset từ file đã xử lý
processed_file = "processed_iot23.pkl"  # hoặc dùng glob nếu muốn linh hoạt hơn
if not os.path.exists(processed_file):
    print("Không tìm thấy processed_iot23.pkl → đang dùng file mới nhất...")
    import glob
    files = glob.glob("processed_*.pkl")
    processed_file = max(files, key=os.path.getctime)  # lấy file mới nhất

print(f"[+] Đang load dữ liệu từ: {processed_file}")
X_train, X_test, y_train, y_test, feature_names = joblib.load(processed_file)

# Phát hiện số class
n_classes = len(np.unique(y_test))
average_mode = 'weighted' if n_classes > 2 else 'binary'
print(f"[+] Phát hiện {n_classes} class → dùng average='{average_mode}'")

# Danh sách model đúng như paper
models = {
    "MLP_DNN":          MLPClassifier(hidden_layer_sizes=(100,50), max_iter=500, random_state=42),
    "GaussianNB":       GaussianNB(),
    "BernoulliNB":      BernoulliNB(),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "SGDClassifier":    SGDClassifier(max_iter=1000, random_state=42),
    "RandomForest":     RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
    "XGBoost":          xgb.XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1),
    "DecisionTree":     DecisionTreeClassifier(random_state=42),
}

results = []
best_f1 = 0
best_model = None
best_name = ""

print(f"\n[+] Bắt đầu huấn luyện {len(models)} model...\n")

for name, model in models.items():
    print(f"Đang train {name}...", end=" ")
    
    # Xử lý riêng XGBoost cho multi-class
    if n_classes > 2 and name == "XGBoost":
        model.set_params(objective='multi:softprob', num_class=n_classes)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    f1 = f1_score(y_test, y_pred, average=average_mode)
    print(f"F1-score = {f1:.4f}")
    
    results.append({"Model": name, "F1-score": round(f1, 4)})
    
    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_name = name

# Lưu kết quả
os.makedirs("results", exist_ok=True)
pd.DataFrame(results).to_csv("results/iot23_results.csv", index=False)
joblib.dump(best_model, "models/best_model_iot23.pkl")

print("\n" + "="*60)
print(f"HOÀN TẤT! BEST MODEL: {best_name}")
print(f"F1-score cao nhất: {best_f1:.4f}")
print("="*60)
print("Kết quả đã lưu:")
print("   → results/iot23_results.csv")
print("   → models/best_model_iot23.pkl")