from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def load_ionosphere(seed=42, test_size=0.2):
    data = fetch_openml("ionosphere", version=1, as_frame=False)
    X = data.data.astype(float)
    y = (data.target == "g").astype(int)   # b (bad return) = 0, g (good return) = 1
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)