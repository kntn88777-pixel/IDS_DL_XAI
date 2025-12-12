# ==================== TIỀN XỬ LÝ DỮ LIỆU (HOÀN CHỈNH) ====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder

import os # Thêm thư viện os để kiểm tra đường dẫn

# ⚠️ LƯU Ý QUAN TRỌNG:
# 1. Thay đổi đường dẫn 'csv_file' và 'output_csv' cho phù hợp với máy tính của bạn.
# 2. Đảm bảo bạn đã cài đặt đủ các thư viện:
#    pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn

# =========================================================================

# 1️⃣ Đọc dữ liệu
# Vui lòng thay đổi đường dẫn này sang file CSV của bạn
csv_file = r"D:\\nienluan\data\\iot23.csv"  

# Kiểm tra sự tồn tại của file trước khi đọc
if not os.path.exists(csv_file):
    print(f"❌ Lỗi: Không tìm thấy file tại đường dẫn: {csv_file}")
    print("Vui lòng cập nhật biến 'csv_file' với đường dẫn chính xác.")
else:
    print("🔹 Đang đọc dữ liệu...")
    try:
        # Sử dụng low_memory=False cho các file lớn, tránh cảnh báo dtype
        df = pd.read_csv(csv_file, low_memory=False)
        
        # In thông tin cơ bản về file
        so_dong_goc = df.shape[0]
        so_cot_goc = df.shape[1]
        print(f"✅ Đọc thành công: {so_dong_goc} dòng, {so_cot_goc} cột (hay {so_cot_goc} đặc trưng)")

        # In tên các đặc trưng (cột)
        print("\n**Danh sách các đặc trưng (cột):**")
        print(df.columns.tolist())
        print("-" * 50)
        
        # 2️⃣ Làm sạch dữ liệu
        print("🔹 Làm sạch dữ liệu...")
        # Xóa các dòng có giá trị thiếu (NaN)
        df.dropna(inplace=True) 
        # Xóa các dòng trùng lặp
        df.drop_duplicates(inplace=True) 
        # Đặt lại chỉ mục sau khi xóa dòng
        df = df.reset_index(drop=True) 
        
        so_dong_sach = df.shape[0]
        so_cot_sach = df.shape[1]
        print(f"✅ Sau làm sạch: {so_dong_sach} dòng, {so_cot_sach} cột")

        # 3️⃣ Xác định cột nhãn
        # Thử tìm cột tên 'label', nếu không có thì lấy cột cuối cùng
        label_col = 'label' if 'label' in df.columns else df.columns[-1]
        print(f"🔹 Cột nhãn được sử dụng: **{label_col}**")

        # 4️⃣ Mã hóa tất cả cột không phải số (Categorical/Object)
        print("🔹 Mã hóa các cột không phải số...")
        for col in df.columns:
            # Kiểm tra kiểu dữ liệu là 'object' (thường là string/category trong pandas)
            if df[col].dtype == 'object':
                le = LabelEncoder()
                # Chuyển sang string trước khi mã hóa để đảm bảo LabelEncoder hoạt động
                df[col] = le.fit_transform(df[col].astype(str)) 
        print("✅ Hoàn tất mã hóa LabelEncoder")

        # 5️⃣ Tách dữ liệu & nhãn
        # X là các đặc trưng (features), y là nhãn (target)
        X = df.drop(columns=[label_col])
        y = df[label_col]
        print(f"🔹 Đặc trưng X có {X.shape[1]} cột (đặc trưng), Nhãn y có {y.nunique()} lớp.")

        # 6️⃣ Chuẩn hóa dữ liệu số (Chỉ áp dụng cho X)
        print("🔹 Chuẩn hóa dữ liệu bằng StandardScaler...")
        scaler = StandardScaler()
        # Áp dụng StandardScaler cho toàn bộ dữ liệu đặc trưng X
        X_scaled = scaler.fit_transform(X) 
        print("✅ Chuẩn hóa hoàn tất")

        # 7️⃣ Vẽ biểu đồ trước cân bằng
        plt.figure(figsize=(7, 4))
        # Sử dụng countplot để xem phân bố của nhãn y
        sns.countplot(x=y) 
        plt.title("📊 Phân bố nhãn (trước khi cân bằng)")
        # Thêm hiển thị giá trị trên các cột
        for container in plt.gca().containers:
            plt.bar_label(container)
        plt.show()

        # 8️⃣ Cân bằng dữ liệu bằng SMOTE (Oversampling)
        print("🔹 Cân bằng dữ liệu bằng SMOTE...")
        # Sử dụng SMOTE để tạo ra các mẫu mới cho các lớp thiểu số
        smote = SMOTE(sampling_strategy='auto', random_state=42) 
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
        print(f"✅ Sau cân bằng: {X_resampled.shape[0]} dòng")

        # 9️⃣ Vẽ biểu đồ sau cân bằng
        plt.figure(figsize=(7, 4))
        sns.countplot(x=y_resampled)
        plt.title("📊 Phân bố nhãn (sau khi cân bằng)")
        # Thêm hiển thị giá trị trên các cột
        for container in plt.gca().containers:
            plt.bar_label(container)
        plt.show()

        # 🔟 Gộp lại thành DataFrame
        # Chuyển dữ liệu đã chuẩn hóa và cân bằng trở lại thành DataFrame
        df_balanced = pd.DataFrame(X_resampled, columns=X.columns)
        df_balanced[label_col] = y_resampled # Thêm cột nhãn đã cân bằng

        # 1️⃣1️⃣ Xuất ra file CSV (lưu toàn bộ)
        # Vui lòng thay đổi đường dẫn này sang nơi bạn muốn lưu file
        output_csv = r"D:\\nienluancoso\data\\iot23_ba.csv" 
        
        # Tạo thư mục nếu nó chưa tồn tại
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df_balanced.to_csv(output_csv, index=False)
        print(f"✅ Đã lưu dữ liệu cân bằng vào: {output_csv}")

        # 1️⃣2️⃣ (Tuỳ chọn) Lưu mẫu nhỏ để xem bằng Excel
        sample_excel = r"D:\\nienluancoso\data\\iot23_balan.xlsx"
        
        # Tạo thư mục nếu nó chưa tồn tại
        os.makedirs(os.path.dirname(sample_excel), exist_ok=True)
        
        # Lấy mẫu ngẫu nhiên 500,000 dòng
        if df_balanced.shape[0] >= 500_000:
            df_sample = df_balanced.sample(n=500_000, random_state=42)
        else:
            # Nếu ít hơn 500k dòng, lưu toàn bộ
            df_sample = df_balanced
            print("⚠️ Cảnh báo: Số dòng sau cân bằng ít hơn 500k, lưu toàn bộ vào file Excel mẫu.")

        df_sample.to_excel(sample_excel, index=False)
        print(f"✅ Đã lưu mẫu nhỏ ({df_sample.shape[0]} dòng) vào: {sample_excel}")

        print("\n🎯 **HOÀN TẤT TIỀN XỬ LÝ DỮ LIỆU**")

    except Exception as e:
        print(f"❌ Đã xảy ra lỗi trong quá trình xử lý: {e}")