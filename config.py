# config.py — thêm 1 dòng thôi!
class Config:
    DATASET_PATH = "data/iot23.csv"
    DATASET_NAME = "iot23"
    LABEL_COLUMN = "label"          # hoặc "detailed-label" tùy file của bạn
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    OVERSAMPLE = True

    # QUAN TRỌNG: Giới hạn mẫu để không crash
    MAX_PER_CLASS = 80_000          # ← 80k/class → tổng ~1 triệu dòng → RAM 8GB vẫn ngon
    # Nếu máy bạn mạnh (32GB+ RAM) → có thể để 150_000