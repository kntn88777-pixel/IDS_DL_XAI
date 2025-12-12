import numpy as np
import pandas as pd
from keras.models import load_model
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import random

# ✅ 1. Đọc dữ liệu và model
X_test = pd.read_csv(r"D:\nienluan\data\X_test.csv")
y_test = pd.read_csv(r"D:\nienluan\data\y_test.csv")
model = load_model(r"D:\nienluan\XDLTDS\RNN.h5")

print(f"✅ Dữ liệu test: {X_test.shape}, mô hình: {model.name}")

# ✅ 2. Hàm dự đoán cho RNN (chuyển 2D → 3D)
def rnn_predict(x):
    x_reshaped = x.reshape((x.shape[0], 1, x.shape[1]))
    return model.predict(x_reshaped)

# ✅ 3. Khởi tạo LIME explainer
explainer = LimeTabularExplainer(
    training_data=np.array(X_test),
    mode="regression",   # dùng regression nếu mô hình trả giá trị liên tục
    feature_names=X_test.columns.tolist(),
    discretize_continuous=True
)

# ✅ 4. Hàm tạo lời giải thích tự nhiên
def generate_text_explanation(explanation):
    top_features = sorted(explanation, key=lambda x: abs(x[1]), reverse=True)[:3]
    desc_lines = []
    for feature, weight in top_features:
        if weight > 0:
            desc_lines.append(f"Giá trị cao của '{feature}' làm tăng khả năng tấn công.")
        else:
            desc_lines.append(f"Giá trị thấp của '{feature}' giúp giảm nguy cơ tấn công.")
    summary = " ".join(desc_lines)
    return f"Theo LIME, các yếu tố quan trọng nhất gồm: {', '.join([f for f, _ in top_features])}. {summary}"

# ✅ 5. Giải thích và xuất PDF
indices = random.sample(range(len(X_test)), 5)  # chọn 5 mẫu ngẫu nhiên
output_pdf = r"D:\nienluan\LIME_Report.pdf"

with PdfPages(output_pdf) as pdf:
    for idx in indices:
        # Giải thích 1 mẫu
        exp = explainer.explain_instance(
            data_row=X_test.iloc[idx],
            predict_fn=rnn_predict,
            num_features=10
        )
        explanation = exp.as_list()
        text_summary = generate_text_explanation(explanation)

        # ✅ Tạo biểu đồ trực tiếp từ LIME
        fig = exp.as_pyplot_figure()
        plt.suptitle(f"LIME Explanation for Sample #{idx}", fontsize=14, y=1.02)

        # ✅ Thêm lời giải thích bằng văn vào cuối biểu đồ
        plt.figtext(
            0.01, -0.1,
            "📘 Giải thích chi tiết:\n" +
            "\n".join([f"- {f}: {w:.4f} ({'↑ Tăng' if w > 0 else '↓ Giảm'})" for f, w in explanation]) +
            "\n\n🗒️ " + text_summary,
            ha="left", va="top", fontsize=9, wrap=True
        )

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

print(f"✅ Báo cáo LIME đầy đủ (có biểu đồ + lời giải thích) đã được tạo tại: {output_pdf}")
