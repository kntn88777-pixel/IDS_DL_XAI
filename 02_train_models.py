# 02_train_models.py — CHỈ DÙNG: DNN, CNN-1D, LSTM, RNN, SVM (NEURAL STYLE)
import joblib
import numpy as np
import pandas as pd
import os
import glob
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM, SimpleRNN, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import save_model

# ========================= TẢI DỮ LIỆU =========================
processed_file = "processed_iot23.pkl"
if not os.path.exists(processed_file):
    print("Không tìm thấy processed_iot23.pkl → lấy file mới nhất...")
    files = glob.glob("processed_*.pkl")
    processed_file = max(files, key=os.path.getctime)

print(f"[+] Đang load dữ liệu từ: {processed_file}")
X_train, X_test, y_train, y_test, feature_names = joblib.load(processed_file)

# Encode nhãn
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

n_classes = len(le.classes_)
input_dim = X_train.shape[1]
average_mode = 'weighted' if n_classes > 2 else 'binary'

print(f"[+] Phát hiện {n_classes} class → average='{average_mode}'")
print(f"[+] Số đặc trưng: {input_dim}")

# Reshape cho CNN/LSTM/RNN: (samples, timesteps=features, channels=1)
X_train_cnn = X_train.astype('float32')[..., np.newaxis]   # (n, features, 1)
X_test_cnn = X_test.astype('float32')[... , np.newaxis]

# Dữ liệu gốc cho DNN
X_train_dnn = X_train.astype('float32')
X_test_dnn = X_test.astype('float32')

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ========================= ĐỊNH NGHĨA 5 MÔ HÌNH HỌC SÂU =========================
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
        RNN(64, return_sequences=False),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(n_classes, activation='softmax' if n_classes > 2 else 'sigmoid')
    ])
    return model

def create_neural_svm():
    # Mạng sâu + output linear → gần giống SVM với hinge loss
    inputs = Input(shape=(input_dim,))
    x = Dense(256, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    outputs = Dense(n_classes, activation='linear')(x)  # linear để gần với SVM
    model = Model(inputs, outputs)
    return model

# Danh sách model đúng yêu cầu của bạn
models = {
    "DNN":         create_dnn(),
    "CNN_1D":      create_cnn_1d(),
    "LSTM":        create_lstm(),
    "RNN":         create_rnn(),
    "Neural_SVM":  create_neural_svm(),
}

results = []
best_f1 = 0
best_model = None
best_name = ""

print(f"\n[+] Bắt đầu huấn luyện 5 mô hình học sâu: DNN, CNN, LSTM, RNN, Neural-SVM...\n")

for name, model in models.items():
    print(f"Đang train {name:<12}...", end=" ")

    # Compile với loss phù hợp
    if name == "Neural_SVM":
        loss = 'hinge' if n_classes == 2 else 'squared_hinge'
        final_activation = 'tanh' if n_classes == 2 else None
        # Áp dụng activation cuối nếu cần
        if n_classes > 2:
            from tensorflow.keras.layers import Activation
            new_output = Activation('softmax')(model.output)
            model = Model(model.input, new_output)
    else:
        loss = 'sparse_categorical_crossentropy' if n_classes > 2 else 'binary_crossentropy'

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=loss,
        metrics=['accuracy']
    )

    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(factor=0.5, patience=7, verbose=0)
    ]

    # Chọn dữ liệu phù hợp
    if name in ["CNN_1D", "LSTM", "RNN"]:
        X_tr = X_train_cnn
        X_te = X_test_cnn
    else:
        X_tr = X_train_dnn
        X_te = X_test_dnn

    model.fit(
        X_tr, y_train_enc,
        validation_data=(X_te, y_test_enc),
        epochs=300,
        batch_size=512,
        callbacks=callbacks,
        verbose=0
    )

    # Dự đoán
    if n_classes > 2:
        pred_probs = model.predict(X_te, verbose=0)
        y_pred = np.argmax(pred_probs, axis=1)
    else:
        pred_probs = model.predict(X_te, verbose=0)
        y_pred = (pred_probs > 0.5).astype(int).flatten() if name != "Neural_SVM" else (pred_probs > 0).astype(int).flatten()

    f1 = f1_score(y_test_enc, y_pred, average=average_mode)
    print(f"F1-score = {f1:.5f}")

    results.append({"Model": name, "F1-score": round(f1, 5)})

    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_name = name

# ========================= LƯU KẾT QUẢ =========================
pd.DataFrame(results).sort_values("F1-score", ascending=False).to_csv("results/iot23_results.csv", index=False)
save_model(best_model, "models/best_model_iot23.h5")

print("\n" + "="*70)
print(f"           HOÀN TẤT! CHỈ DÙNG 5 MÔ HÌNH HỌC SÂU")
print("="*70)
print(f"   BEST MODEL     → {best_name}")
print(f"   F1-score cao nhất → {best_f1:.5f}")
print("="*70)
for r in results:
    print(f"   {r['Model']:<12} → F1 = {r['F1-score']}")
print("="*70)
print("Kết quả lưu tại:")
print("   → results/iot23_results.csv")
print("   models/best_model_iot23.h5")
print("="*70)