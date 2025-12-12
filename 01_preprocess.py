# 01_preprocess.py
from config import Config
from utils import load_and_preprocess
import joblib

if __name__ == "__main__":
    config = Config()
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess(config)

    joblib.dump((X_train, X_test, y_train, y_test, feature_names), "processed_data.pkl")
    print("[+] Đã lưu dữ liệu đã xử lý → processed_data.pkl")