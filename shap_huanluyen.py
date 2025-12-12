import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ================== 1️⃣ Đọc dữ liệu và mô hình ==================
X_test = pd.read_csv(r"D:\\nienluan\data\X_test.csv")
y_test = pd.read_csv(r"D:\\nienluan\data\y_test.csv").squeeze()
model = load_model(r"D:\\nienluan\data\RNN.h5")

# ================== 2️⃣ Lấy mẫu nhỏ ==================
SAMPLE_SIZE = 200
X_sample = X_test.sample(SAMPLE_SIZE, random_state=42)
print(f"Using {SAMPLE_SIZE} samples")

# ================== 3️⃣ KernelExplainer ==================
def model_predict(x):
    x_rnn = np.expand_dims(x, axis=1)
    return model.predict(x_rnn, verbose=0)

explainer = shap.KernelExplainer(model_predict, X_sample.iloc[:50])
shap_values_list = explainer.shap_values(X_sample)  # list of arrays

# ================== 4️⃣ Chuẩn bị DataFrame ==================
# Nếu multi-class → tính trung bình
shap_values_mean = np.mean(np.array(shap_values_list), axis=0)
X_sample_df = X_sample.reset_index(drop=True)  # giữ tên cột

# ================== 5️⃣ Vẽ summary plot ==================
plt.figure(figsize=(10,6))
shap.summary_plot(shap_values_mean, X_sample_df, show=False, plot_type="bar")
plt.tight_layout()
plt.savefig("shap_feature_importance.png")
plt.close()
print("Saved shap_feature_importance.png")

# ================== 6️⃣ Force plot mẫu ==================
sample_idx = 0
force_plot = shap.force_plot(
    explainer.expected_value[0],
    shap_values_list[0][sample_idx],
    X_sample_df.iloc[sample_idx],
    matplotlib=True
)
plt.savefig("shap_force_sample0.png")
plt.close()
print("Saved shap_force_sample0.png")
