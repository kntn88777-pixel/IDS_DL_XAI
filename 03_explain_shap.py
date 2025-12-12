import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
import os

# Tự động tìm file processed và best_model mới nhất
processed_files = [f for f in os.listdir('.') if f.startswith('processed_') and f.endswith('.pkl')]
if not processed_files:
    print("Không tìm thấy file processed_*.pkl → chạy lại 01_preprocess.py")
    exit()
processed_file = max(processed_files, key=os.path.getctime)

model_files = [f for f in os.listdir('models') if f.startswith('best_model_') and f.endswith('.pkl')]
if not model_files:
    print("Không tìm thấy best_model_*.pkl → chạy lại 02_train_models.py")
    exit()
model_file = model_files[-1]  # lấy file mới nhất

print(f"Đang load dữ liệu: {processed_file}")
print(f"Best model: models/{model_file}")

# Load dữ liệu và model
X_train, X_test, y_train, y_test, feature_names = joblib.load(processed_file)
model = joblib.load(f"models/{model_file}")

# Chỉ lấy 150 mẫu để vẽ SHAP 
X_sample = shap.sample(X_test, 150)
print(f"Đang tính SHAP values cho {len(X_sample)} ")

# Tự động chọn explainer phù hợp
if any(keyword in str(type(model)) for keyword in ["XGB", "RandomForest", "LGBM", "CatBoost", "GradientBoosting"]):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
else:
    # Dùng KernelExplainer cho các model khác (MLP, LR, v.v.)
    background = shap.sample(X_train, 50)
    explainer = shap.KernelExplainer(model.predict_proba, background)
    shap_values = explainer.shap_values(X_sample[:30])  # chỉ 30 mẫu để nhanh

os.makedirs("plots", exist_ok=True)

# Vẽ Summary Plot (đẹp nhất, giống Fig. 8 trong paper)
plt.figure(figsize=(10, 7))
if isinstance(shap_values, list):  # multi-class
    for i, sv in enumerate(shap_values):
        shap.summary_plot(sv, X_sample, feature_names=feature_names, show=False, max_display=15)
        plt.title(f"SHAP Summary - Class {i}")
        plt.savefig(f"plots/SHAP_summary_class_{i}.png", dpi=300, bbox_inches='tight')
        plt.close()
else:
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False, max_display=15)
    plt.title("SHAP Summary Plot")
    plt.savefig("plots/SHAP_summary.png", dpi=300, bbox_inches='tight')
    plt.close()

print("HOÀN TẤT! SHAP plots đã lưu:")
print("   → plots/SHAP_summary.png (hoặc SHAP_summary_class_*.png)")