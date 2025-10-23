import os

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

for path in [DATA_DIR, ARTIFACTS_DIR]:
    os.makedirs(path, exist_ok=True)