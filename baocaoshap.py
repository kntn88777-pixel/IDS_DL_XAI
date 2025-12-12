# xai_analysis_with_grok.py
import shap
import lime
from lime import lime_tabular
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from tensorflow.keras.models import load_model
import os

# ================== CẤU HÌNH ==================
DATA_PATH = r"D:\nienluan\data"
MODEL_PATH = os.path.join(DATA_PATH, "RNN.h5")
X_TEST_PATH = os.path.join(DATA_PATH, "X_test.csv")
Y_TEST_PATH = os.path.join(DATA_PATH, "y_test.csv")

# Grok-4 API (thay key thật của bạn)
GROK_API_KEY = "gsk_your_key_here"
GROK_API_URL = "https://api.x.ai/v1/chat/completions"

# ================== 1. ĐỌC DỮ LIỆU & MÔ HÌNH ==================
print("Đang đọc dữ liệu...")
X_test = pd.read_csv(X_TEST_PATH)
y_test = pd.read_csv(Y_TEST_PATH).squeeze() if isinstance(pd.read_csv(Y_TEST_PATH), pd.DataFrame) else pd.read_csv(Y_TEST_PATH).iloc[:, 0]

model = load_model(MODEL_PATH)
print(f"Model: {MODEL_PATH} – Đã load thành công!")
print(f"X_test shape: {X_test.shape}, Số lớp: {len(np.unique(y_test))}")

# ================== 2. CHUẨN BỊ BACKGROUND CHO SHAP & LIME ==================
SAMPLE_SIZE = 100  # Giảm để nhanh (KernelExplainer rất chậm)
background = X_test.sample(SAMPLE_SIZE, random_state=42).values

# Hàm dự đoán đúng shape cho RNN (batch, timesteps=1, features)
def predict_fn(data):
    data = np.array(data)
    data_reshaped = data.reshape(data.shape[0], 1, data.shape[1])  # (samples, 1, features)
    return model.predict(data_reshaped, verbose=0)

# ================== 3. SHAP – DÙNG KernelExplainer (ổn định nhất) ==================
print("Đang tính SHAP (có thể mất 5-15 phút với 100 mẫu)...")
explainer_shap = shap.KernelExplainer(predict_fn, background[:50])  # 50 mẫu background

# Chỉ lấy 20 mẫu để test nhanh
X_explain = X_test.sample(20, random_state=42).values
shap_values = explainer_shap.shap_values(X_explain)

# Nếu multi-class → shap_values là list → lấy class 1 (tấn công)
if isinstance(shap_values, list):
    shap_values = shap_values[1]  # class 1 = tấn công

print("SHAP hoàn tất!")

# ================== 4. LIME ==================
print("Đang tính LIME...")
explainer_lime = lime_tabular.LimeTabularExplainer(
    training_data=background,
    feature_names=X_test.columns.tolist(),
    class_names=['Benign', 'Attack'],
    mode='classification'
)

lime_explanations = []
for i in range(min(10, len(X_explain))):
    exp = explainer_lime.explain_instance(
        data_row=X_explain[i],
        predict_fn=predict_fn,
        num_features=10
    )
    lime_explanations.append(exp.as_list())

# ================== 5. VẼ SHAP PLOT ==================
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_explain, feature_names=X_test.columns, show=False)
plt.tight_layout()
plt.savefig("shap_summary_attack.png", dpi=200, bbox_inches='tight')
plt.close()
print("Đã lưu: shap_summary_attack.png")

# Force plot mẫu đầu tiên
shap.initjs()
force_html = shap.force_plot(
    explainer_shap.expected_value[1] if isinstance(explainer_shap.expected_value, list) else explainer_shap.expected_value,
    shap_values[0],
    X_explain[0],
    feature_names=X_test.columns,
    matplotlib=False
)
shap.save_html("shap_force_sample0.html", force_html)
print("Đã lưu: shap_force_sample0.html")

# ================== 6. GỌI GROK-4 TỰ ĐỘNG SINH BÁO CÁO ==================
def call_grok(prompt):
    headers = {"Authorization": f"Bearer {GROK_API_KEY}"}
    payload = {
        "model": "grok-beta",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }
    try:
        r = requests.post(GROK_API_URL, json=payload, headers=headers, timeout=30)
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Lỗi gọi Grok: {e}"

# Top features từ SHAP
shap_df = pd.DataFrame(shap_values, columns=X_test.columns)
top_features = shap_df.abs().mean().sort_values(ascending=False).head(10)

prompt = f"""
Phân tích XAI trên mô hình RNN phát hiện tấn công mạng (dữ liệu CIC-IDS2017 hoặc tương tự).

Top 10 tính năng quan trọng nhất (SHAP value trung bình):
{top_features.to_string()}

LIME top 5 (mẫu đầu tiên):
{lime_explanations[0][:5] if lime_explanations else "N/A"}

Yêu cầu:
- Viết báo cáo tiếng Việt chuyên nghiệp
- Giải thích tại sao các feature này quan trọng
- Gợi ý giải pháp chặn tấn công (chặn IP, rate-limit, WAF...)
- Định dạng Markdown đẹp, có tiêu đề, bảng, danh sách
"""

print("Đang gọi Grok-4 để sinh báo cáo tự động...")
report = call_grok(prompt)

# Lưu báo cáo
with open("BÁO_CÁO_XAI_TỰ_ĐỘNG.md", "w", encoding="utf-8") as f:
    f.write("# BÁO CÁO XAI TỰ ĐỘNG – RNN IDS\n\n")
    f.write(report)

print("\n" + "="*60)
print("HOÀN TẤT! ĐÃ TẠO:")
print("   shap_summary_attack.png")
print("   shap_force_sample0.html")
print("   BÁO_CÁO_XAI_TỰ_ĐỘNG.md  ← MỞ LÊN LÀ CÓ BÁO CÁO ĐẸP!")
print("="*60)