import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, LSTM, SimpleRNN
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

# ================== 🔹 CẤU HÌNH ĐƯỜNG DẪN ==================
base_path = r"D:\\nienluan\data"  # chỉnh lại nếu cần
X_train_path = os.path.join(base_path, "X_train.csv")
X_test_path  = os.path.join(base_path, "X_test.csv")
y_train_path = os.path.join(base_path, "y_train.csv")
y_test_path  = os.path.join(base_path, "y_test.csv")

# ================== 🔹 ĐỌC DỮ LIỆU ==================
print("📂 Đang đọc dữ liệu...")
X_train = pd.read_csv(X_train_path)
X_test  = pd.read_csv(X_test_path)
y_train = pd.read_csv(y_train_path).squeeze()  # squeeze để chuyển thành Series
y_test  = pd.read_csv(y_test_path).squeeze()

print(f"✅ Dữ liệu đọc thành công:")
print(f"   X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"   y_train: {y_train.shape}, y_test: {y_test.shape}")

n_features = X_train.shape[1]
n_classes = len(np.unique(y_train))

# ================== 🔹 HÀM ĐÁNH GIÁ ==================
def evaluate_model(model, X_test, y_test, name="Model"):
    y_pred = np.argmax(model.predict(X_test), axis=1)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🔹 {name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ================== 🔹 HÀM HUẤN LUYỆN CHUNG ==================
def train_model(model, X_train, y_train, X_test, y_test, name="Model"):
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.compile(optimizer=Adam(0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, batch_size=256, validation_data=(X_test, y_test), callbacks=[es], verbose=2)
    model.save(os.path.join(base_path, f"{name}.h5"))
    evaluate_model(model, X_test, y_test, name)

# ================== 🔹 MÔ HÌNH DNN ==================
print("\n🚀 Huấn luyện mô hình DNN...")
model_dnn = Sequential([
    Dense(256, activation='relu', input_shape=(n_features,)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(n_classes, activation='softmax')
])
train_model(model_dnn, X_train, y_train, X_test, y_test, "DNN")

# ================== 🔹 MÔ HÌNH CNN ==================
print("\n🚀 Huấn luyện mô hình CNN...")
X_train_cnn = np.expand_dims(X_train, axis=2)
X_test_cnn = np.expand_dims(X_test, axis=2)
model_cnn = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(n_features,1)),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    Conv1D(128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(n_classes, activation='softmax')
])
train_model(model_cnn, X_train_cnn, y_train, X_test_cnn, y_test, "CNN")

# ================== 🔹 MÔ HÌNH RNN ==================
print("\n🚀 Huấn luyện mô hình RNN...")
X_train_rnn = np.expand_dims(X_train, axis=1)
X_test_rnn = np.expand_dims(X_test, axis=1)
model_rnn = Sequential([
    SimpleRNN(128, activation='tanh', input_shape=(1, n_features)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(n_classes, activation='softmax')
])
train_model(model_rnn, X_train_rnn, y_train, X_test_rnn, y_test, "RNN")

# ================== 🔹 MÔ HÌNH LSTM ==================
print("\n🚀 Huấn luyện mô hình LSTM...")
X_train_lstm = np.expand_dims(X_train, axis=1)
X_test_lstm = np.expand_dims(X_test, axis=1)
model_lstm = Sequential([
    LSTM(128, input_shape=(1, n_features)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(n_classes, activation='softmax')
])
train_model(model_lstm, X_train_lstm, y_train, X_test_lstm, y_test, "LSTM")

print("\n🎯 Hoàn tất huấn luyện. Các mô hình đã lưu trong thư mục XDLTDS/")
