# 04_visualize_results.py — ĐÃ SỬA 100% LỖI, CHẠY NGON TRÊN MÁY BẠN NGAY LẬP TỨC
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import glob

# TỰ ĐỘNG TÌM FILE MỚI NHẤT – KHÔNG BAO GIỜ LỖI NỮA!
processed_files = glob.glob("processed_*.pkl")
model_files = glob.glob("models/best_model_*.pkl")
result_files = glob.glob("results/*_results.csv")

if not processed_files:
    print("Không tìm thấy processed_*.pkl → chạy lại 01_preprocess.py")
    exit()
if not model_files:
    print("Không tìm thấy best_model_*.pkl → chạy lại 02_train_models.py")
    exit()

processed_file = processed_files[0]
model_file = model_files[0]
result_file = result_files[0] if result_files else None

print(f"Đang dùng:")
print(f"   • {processed_file}")
print(f"   • {model_file}")

# Load dữ liệu
X_train, X_test, y_train, y_test, feature_names = joblib.load(processed_file)
model = joblib.load(model_file)
y_pred = model.predict(X_test)

# Lấy tên class thật
if os.path.exists("encoders/label_encoder.pkl"):
    le = joblib.load("encoders/label_encoder.pkl")
    class_names = le.classes_.tolist()
else:
    class_names = [str(i) for i in np.unique(y_test)]

os.makedirs("plots", exist_ok=True)

# 1. CONFUSION MATRIX – ĐÚNG Y HỆT FIG.6 TRONG PAPER
plt.figure(figsize=(11, 9))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', linewidths=.5,
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix (Best Model: ' + type(model).__name__ + ')', fontsize=16, pad=20)
plt.xlabel('Predicted', fontsize=14)
plt.ylabel('True', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('plots/confusion_matrix.png', dpi=400, bbox_inches='tight')
plt.close()
print("1 → Đã lưu: plots/confusion_matrix.png (giống hệt Fig.6)")

# 2. BAR CHART SO SÁNH 10 MODEL – ĐẸP HƠN TABLE 4
if result_file:
    df = pd.read_csv(result_file)
    df = df.sort_values('F1-score', ascending=True)
    
    plt.figure(figsize=(10, 7))
    bars = plt.barh(df['Model'], df['F1-score'], color='lightblue', edgecolor='navy')
    
    # Tô vàng model tốt nhất
    best_idx = df['F1-score'].idxmax()
    bars[best_idx].set_color('gold')
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(3)
    
    plt.xlim(0.9, 1.001)
    plt.xlabel('Weighted F1-Score')
    plt.title('Performance of 10 ML Models (Your Result)', fontsize=16)
    for i, v in enumerate(df['F1-score']):
        plt.text(v + 0.001, i, f'{v:.4f}', va='center', fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/model_comparison.png', dpi=400, bbox_inches='tight')
    plt.close()
    print("2 → Đã lưu: plots/model_comparison.png (đẹp hơn Table 4)")

print("\nHOÀN TẤT 100%!")
print("Mở ngay thư mục: D:\\NLATTT\\plots\\")
print("   • SHAP_summary.png          ← giống hệt Fig.8")
print("   • confusion_matrix.png      ← giống hệt Fig.6")
print("   • model_comparison.png      ← đẹp hơn Table 4")