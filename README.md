# IDS_DL_XAI tất cả các src code do chatgpt sinh tự động từ yêu cầu Tài liệu tham khảo từ nguồn do các bài báo và do chatgpt sinh tự động từ yêu cầu.
Hướng Dẫn Thực Hiện Đề Tài: Ứng Dụng xAI Trong Tăng Cường Khả Năng Phân Tích Hiệu Quả Của Các Mô Hình Học Sâu Trên Hệ Thống Phát Hiện Xâm Nhập
Dựa trên tài liệu luận văn được cung cấp (file 4.docx), tôi sẽ viết lại toàn bộ các bước thực hiện đề tài một cách chi tiết, có cấu trúc rõ ràng. Các bước được lấy cảm hứng từ các chương trong luận văn, tập trung vào phương pháp nghiên cứu, thực hiện, và kết quả. Tất cả source code liên quan được trích xuất và viết lại từ các đoạn mã trong tài liệu (Chương 3), với giả định rằng toàn bộ mã nguồn đầy đủ được lưu trữ trên GitHub tại: https://github.com/kntn88777-pixel/IDS_DL_XAI. Repo này chứa các file Python (.py) cho việc tiền xử lý dữ liệu, huấn luyện mô hình, áp dụng SHAP, và tạo báo cáo. Nếu repo có notebook (.ipynb), chúng có thể được sử dụng để chạy tương tác.
Lưu ý:

Môi trường thực nghiệm: Windows 11 64-bit, Python với thư viện như NumPy, Pandas, TensorFlow/Keras, SHAP, Scikit-learn, Joblib, Matplotlib, Glob, OS.
Dữ liệu: Sử dụng các bộ dữ liệu công khai như IoT23 (processed_iot23.pkl), CIC-IDS2017/2018, UNSW-NB15. Dữ liệu cần được tải về và lưu ở định dạng pickle (.pkl) hoặc CSV.
Các bước được viết lại để dễ tái hiện, với code được tối ưu hóa nhẹ cho tính rõ ràng (thêm comment).

Bước 1: Thu Thập và Tiền Xử Lý Dữ Liệu (Dựa trên Chương 3: Chuẩn Bị Dữ Liệu)
Mục tiêu: Thu thập dữ liệu từ các bộ dữ liệu công khai (IoT23, CIC-IDS2017, v.v.), làm sạch, chuẩn hóa, và cân bằng lớp để tránh thiên lệch.

Thu thập dữ liệu: Tải bộ dữ liệu IoT23 từ nguồn công khai. Bao gồm lưu lượng mạng bình thường và tấn công (DDoS, SYN Flood, v.v.).
Tiền xử lý:
Làm sạch: Xử lý giá trị thiếu, loại bỏ trùng lặp.
Chuẩn hóa: Min-Max scaling cho đặc trưng số.
Mã hóa: One-hot encoding cho đặc trưng phân loại (e.g., protocol, conn_state).
Cân bằng lớp: Sử dụng SMOTE cho lớp thiểu số (tấn công).

Lưu dữ liệu: Lưu dưới dạng pickle (.pkl) cho dễ load.

Source code (file: data_preprocessing.py trên GitHub):
Pythonimport pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
import os
import glob

# Bước 1: Tải dữ liệu thô (giả sử từ CSV)
def load_raw_data(train_file='train.csv', test_file='test.csv'):
    X_train = pd.read_csv(train_file)
    y_train = pd.read_csv('y_train.csv').squeeze()  # Chuyển DataFrame 1 cột thành Series
    X_test = pd.read_csv(test_file)
    y_test = pd.read_csv('y_test.csv').squeeze()
    return X_train, X_test, y_train, y_test

# Bước 2: Tiền xử lý
def preprocess_data(X_train, X_test, y_train, y_test):
    # Xử lý giá trị thiếu
    X_train.fillna(0, inplace=True)
    X_test.fillna(0, inplace=True)
    
    # Chuẩn hóa Min-Max cho đặc trưng số
    scaler = MinMaxScaler()
    num_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    
    # One-hot encoding cho đặc trưng phân loại
    cat_cols = X_train.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_train_cat = pd.DataFrame(encoder.fit_transform(X_train[cat_cols]))
        X_test_cat = pd.DataFrame(encoder.transform(X_test[cat_cols]))
        X_train = pd.concat([X_train.drop(cat_cols, axis=1), X_train_cat], axis=1)
        X_test = pd.concat([X_test.drop(cat_cols, axis=1), X_test_cat], axis=1)
    
    # Mã hóa nhãn
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    
    # Cân bằng lớp bằng SMOTE (nếu cần)
    smote = SMOTE()
    X_train_bal, y_train_enc_bal = smote.fit_resample(X_train, y_train_enc)
    
    # Lấy thông tin
    n_features = X_train.shape[1]
    n_classes = len(le.classes_)
    feature_names = X_train.columns.tolist()
    
    # Lưu dữ liệu đã xử lý
    processed_file = 'processed_iot23.pkl'
    joblib.dump((X_train_bal, X_test, y_train_enc_bal, y_test_enc, feature_names), processed_file)
    print(f"Dữ liệu đã lưu tại: {processed_file}")
    
    return X_train_bal, X_test, y_train_enc_bal, y_test_enc, n_features, n_classes, feature_names

# Chạy
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_raw_data()
    preprocess_data(X_train, X_test, y_train, y_test)
Bước 2: Xây Dựng và Huấn Luyện Các Mô Hình Học Sâu (Dựa trên Chương 3: Huấn Luyện Các Mô Hình)
Mục tiêu: Xây dựng 5 mô hình (DNN, CNN 1D, LSTM, RNN, Neural-SVM), huấn luyện trên dữ liệu đã xử lý, đánh giá bằng F1-score.

Chuẩn bị: Load dữ liệu từ .pkl, reshape cho mô hình chuỗi (CNN/LSTM/RNN).
Xây dựng mô hình: Sử dụng Keras.
Huấn luyện: Sử dụng Adam optimizer, early stopping.
Đánh giá: Tính F1-score, lưu mô hình tốt nhất.

Source code (file: train_models.py trên GitHub):
Pythonimport tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM, SimpleRNN, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import f1_score
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load dữ liệu đã xử lý
processed_file = "processed_iot23.pkl"
if not os.path.exists(processed_file):
    files = glob.glob("processed_*.pkl")
    processed_file = max(files, key=os.path.getctime) if files else None
    if not processed_file:
        raise FileNotFoundError("Không tìm thấy file dữ liệu!")

X_train, X_test, y_train_enc, y_test_enc, feature_names = joblib.load(processed_file)
n_classes = len(np.unique(y_train_enc))
input_dim = X_train.shape[1]
average_mode = 'weighted' if n_classes > 2 else 'binary'

# Reshape cho mô hình chuỗi
X_train_cnn = X_train.astype('float32')[:, :, np.newaxis] if len(X_train.shape) == 2 else X_train
X_test_cnn = X_test.astype('float32')[:, :, np.newaxis] if len(X_test.shape) == 2 else X_test
X_train_dnn = X_train.astype('float32')
X_test_dnn = X_test.astype('float32')

# Tạo thư mục lưu
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Định nghĩa mô hình
def create_dnn():
    model = Sequential([
        Dense(512, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(n_classes, activation='softmax' if n_classes > 2 else 'sigmoid')
    ])
    return model

def create_cnn_1d():
    model = Sequential([
        Conv1D(128, kernel_size=3, activation='relu', input_shape=(input_dim, 1)),
        BatchNormalization(),
        Conv1D(64, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Conv1D(64, kernel_size=3, activation='relu'),
        BatchNormalization(),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(n_classes, activation='softmax' if n_classes > 2 else 'sigmoid')
    ])
    return model

def create_lstm():
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(input_dim, 1)),
        Dropout(0.4),
        LSTM(64, return_sequences=False),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(n_classes, activation='softmax' if n_classes > 2 else 'sigmoid')
    ])
    return model

def create_rnn():
    model = Sequential([
        SimpleRNN(128, return_sequences=True, input_shape=(input_dim, 1)),
        Dropout(0.4),
        SimpleRNN(64, return_sequences=False),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(n_classes, activation='softmax' if n_classes > 2 else 'sigmoid')
    ])
    return model

def create_neural_svm():
    inputs = Input(shape=(input_dim,))
    x = Dense(256, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    outputs = Dense(n_classes, activation='linear')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Danh sách mô hình
models = {
    "DNN": create_dnn(),
    "CNN_1D": create_cnn_1d(),
    "LSTM": create_lstm(),
    "RNN": create_rnn(),
    "Neural_SVM": create_neural_svm()
}

# Callback chung
callbacks = [
    EarlyStopping(patience=15, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.5, patience=7, verbose=1)
]

# Huấn luyện và đánh giá
results = []
for name, model in models.items():
    # Biên dịch mô hình
    loss = 'sparse_categorical_crossentropy' if n_classes > 2 else 'binary_crossentropy'
    if name == "Neural_SVM":
        loss = 'hinge'  # Cho SVM-like
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    
    # Chọn dữ liệu phù hợp (CNN/LSTM/RNN dùng X_cnn, DNN/SVM dùng X_dnn)
    X_train_use = X_train_cnn if name in ["CNN_1D", "LSTM", "RNN"] else X_train_dnn
    X_test_use = X_test_cnn if name in ["CNN_1D", "LSTM", "RNN"] else X_test_dnn
    
    # Huấn luyện
    model.fit(X_train_use, y_train_enc, epochs=100, batch_size=32, validation_split=0.2, callbacks=callbacks, verbose=1)
    
    # Dự đoán và đánh giá
    y_pred_prob = model.predict(X_test_use)
    y_pred = np.argmax(y_pred_prob, axis=1) if n_classes > 2 else (y_pred_prob > 0.5).astype(int)
    f1 = f1_score(y_test_enc, y_pred, average=average_mode)
    results.append({"Model": name, "F1-score": f1})
    
    # Lưu mô hình
    model.save(f"models/{name}_model.h5")

# Lưu kết quả
pd.DataFrame(results).sort_values("F1-score", ascending=False).to_csv("results/iot23_results.csv", index=False)

# Lưu mô hình tốt nhất
best_model_name = max(results, key=lambda x: x['F1-score'])['Model']
best_model = models[best_model_name]
best_model.save("models/best_model_iot23.h5")
Bước 3: Áp Dụng xAI Với SHAP Để Giải Thích Mô Hình (Dựa trên Chương 3: Xây Dựng X-AI Với SHAP)
Mục tiêu: Sử dụng SHAP để tính giá trị giải thích, vẽ biểu đồ (summary plot, waterfall), và phân tích đặc trưng quan trọng.

Load mô hình và dữ liệu: Từ .pkl và .h5.
Tính SHAP: Sử dụng TreeExplainer (nếu tree-based) hoặc DeepExplainer.
Trực quan hóa: Biểu đồ bar, summary, waterfall, dependence.
Phân tích nhân quả: Lưu vào file txt.

Source code (file: explain_with_shap.py trên GitHub):
Pythonimport shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
from tensorflow.keras.models import load_model

# Tìm file dữ liệu và mô hình
def find_file(patterns):
    for p in patterns:
        files = glob.glob(p, recursive=True)
        if files:
            return max(files, key=os.path.getctime)
    return None

processed_file = find_file(["processed_*.pkl", "**/*.pkl"])
model_file = find_file(["models/best_model_*.h5", "best_model_*.h5", "models/*.h5"])

if not processed_file or not model_file:
    raise FileNotFoundError("Không tìm thấy file dữ liệu hoặc mô hình!")

# Load dữ liệu
data = joblib.load(processed_file)
if isinstance(data, tuple):
    X_test = data[1] if len(data) >= 3 else data[0]
    feature_names = data[-1] if len(data) >= 5 else [f"f{i}" for i in range(X_test.shape[1])]
else:
    X_test = data
    feature_names = [f"f{i}" for i in range(X_test.shape[1])]

X_test = pd.DataFrame(X_test, columns=feature_names)

# Load mô hình (giả sử Keras model)
model = load_model(model_file)

# Lấy mẫu dữ liệu
X_sample = shap.kmeans(X_test, 200).data  # Hoặc shap.sample(X_test, 200)
X_single = X_sample[0:1]  # Mẫu đơn

# Khởi tạo explainer (DeepExplainer cho DL models)
explainer = shap.DeepExplainer(model, X_sample)  # Hoặc TreeExplainer nếu tree-based

# Tính SHAP values
shap_values = explainer.shap_values(X_sample)
shap_single = explainer.shap_values(X_single)

# Xử lý cho binary classification (lấy lớp 1: tấn công)
if isinstance(shap_values, list):
    shap_values = shap_values[1]
    shap_single = shap_single[1]

# Biểu đồ bar: Độ quan trọng toàn cục
shap.summary_plot(shap_values, X_sample, plot_type="bar", feature_names=feature_names, show=False)
plt.savefig("results/shap_bar.png")
plt.close()

# Biểu đồ summary: Phân bố và tác động
shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
plt.savefig("results/shap_summary.png")
plt.close()

# Biểu đồ dependence cho đặc trưng quan trọng (ví dụ: feat_index 0)
feat = feature_names[0]
idx = feature_names.index(feat)
x_vals = X_sample[feat].values
y_vals = shap_values[:, idx]
plt.scatter(x_vals, y_vals, alpha=0.6)
plt.title(f"Dependence Plot for {feat}")
plt.savefig("results/shap_dependence.png")
plt.close()

# Biểu đồ waterfall cho mẫu đơn
expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
shap.waterfall_plot(
    shap.Explanation(
        values=shap_single.flatten(),
        base_values=expected_value,
        data=X_single.flatten(),
        feature_names=feature_names
    ),
    show=False
)
plt.savefig("results/shap_waterfall.png")
plt.close()

# Lưu phân tích nhân quả (ví dụ)
with open("results/causal_graph.txt", "w", encoding="utf-8") as f:
    f.write("flow_duration → Attack\n")
    # Thêm các quan hệ khác dựa trên SHAP
Bước 4: Kết Hợp AI Để Tạo Báo Cáo (Dựa trên Chương 3: Kết Hợp Các AI Có Sẵn)
Mục tiêu: Trích xuất bất thường từ SHAP, áp dụng heuristic để suy luận tấn công, tạo báo cáo HTML/PDF.

Trích xuất: Từ SHAP values, xác định đặc trưng bất thường.
Heuristic: Quy tắc thủ công (e.g., thời gian luồng ngắn → SYN Flood).
Tạo báo cáo: Sử dụng template HTML hoặc AI (như GPT) để mô tả.

Source code (file: generate_report.py trên GitHub):
Pythonimport pandas as pd
import shap
# ... (load SHAP values từ bước trước)

# Trích xuất đặc trưng bất thường
top_features = pd.DataFrame({'Feature': feature_names, 'SHAP_mean': np.mean(np.abs(shap_values), axis=0)})
top_features = top_features.sort_values('SHAP_mean', ascending=False).head(10)

# Heuristic rules (ví dụ)
abnormalities = []
for idx, row in top_features.iterrows():
    feat = row['Feature']
    if 'duration' in feat and row['SHAP_mean'] > 0.5:  # Ngưỡng ví dụ
        abnormalities.append(f"{feat}: Short duration → Possible SYN Flood")
    # Thêm rules khác: SYN flags, packet size, etc.

# Suy luận tổng quát
if len(abnormalities) >= 5:
    conclusion = "SYN Flood kết hợp DDoS volume-based có tổ chức cao"
elif len(abnormalities) >= 3:
    conclusion = "Dấu hiệu SYN Flood rõ ràng"
else:
    conclusion = "Nghi ngờ DoS chung"

# Tạo báo cáo (ví dụ: lưu HTML)
report_html = f"""
<html>
<body>
<h1>Báo Cáo Phân Tích Tấn Công</h1>
<p>Kết luận: {conclusion}</p>
<ul>
{"".join(f"<li>{ab}</li>" for ab in abnormalities)}
</ul>
<p>Đề xuất: Kích hoạt SYN Cookies, giới hạn tốc độ, sử dụng dịch vụ chống DDoS.</p>
</body>
</html>
"""
with open("results/report.html", "w") as f:
    f.write(report_html)
Bước 5: Đánh Giá Kết Quả Và Kết Luận (Dựa trên Chương 4 & 5)

Đánh giá: So sánh F1-score (Bảng 3), vẽ ma trận nhầm lẫn (Hình 6), biểu đồ SHAP (Hình 7).
Kết quả: Độ chính xác 99-99.4%, F1-score >0.99. SHAP xác định đặc trưng quan trọng như id.resp_p, flow_duration.
Hạn chế: Dữ liệu công khai, chi phí tính toán cao.
Hướng phát triển: Tích hợp Transformer, dữ liệu thực tế, automated response.
