# utils.py — PHIÊN BẢN CUỐI CÙNG, CHẠY NGON TẤT CẢ DATASET
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import joblib
import os

def load_and_preprocess(config):
    os.makedirs("encoders", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    print(f"[+] Đang đọc {config.DATASET_NAME} ...")
    df = pd.read_csv(config.DATASET_PATH, low_memory=False)
    print(f"[+] Loaded {config.DATASET_NAME}: {df.shape[0]:,} dòng × {df.shape[1]} cột")

    df = df.fillna('missing')

    # === CỐT LÕI: GIỚI HẠN SỐ MẪU MỖI CLASS ĐỂ TRÁNH HẾT RAM ===
    MAX_PER_CLASS = getattr(config, "MAX_PER_CLASS", 100_000)  # ← MỚI THÊM
    if MAX_PER_CLASS > 0:
        print(f"[+] Giới hạn tối đa {MAX_PER_CLASS:,} mẫu mỗi class để tránh crash")
        df = df.groupby(config.LABEL_COLUMN).apply(lambda x: x.sample(n=min(len(x), MAX_PER_CLASS), random_state=42)).reset_index(drop=True)

    # Encode categorical
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if config.LABEL_COLUMN in cat_cols:
        cat_cols.remove(config.LABEL_COLUMN)

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        joblib.dump(le, f"encoders/encoder_{col}.pkl")

    # Encode label
    if df[config.LABEL_COLUMN].dtype == 'object':
        le_label = LabelEncoder()
        df[config.LABEL_COLUMN] = le_label.fit_transform(df[config.LABEL_COLUMN])
        joblib.dump(le_label, "encoders/label_encoder.pkl")
        n_classes = len(le_label.classes_)
    else:
        n_classes = df[config.LABEL_COLUMN].nunique()

    X = df.drop(columns=[config.LABEL_COLUMN])
    y = df[config.LABEL_COLUMN].values

    # Chia train/test
    class_counts = Counter(y)
    min_count = min(class_counts.values())
    if min_count >= 2:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    else:
        print(f"[!] Có class chỉ có 1 mẫu → bỏ stratify")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Oversample nhẹ nhàng hơn (hoặc tắt nếu cần)
    if config.OVERSAMPLE:
        print(f"[+] Oversampling nhẹ (target = {min(50000, len(y_train)//n_classes)} mẫu/class)")
        ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
        X_train, y_train = ros.fit_resample(X_train, y_train)

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, "models/scaler.pkl")

    print(f"[+] DONE! Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, X.columns.tolist()