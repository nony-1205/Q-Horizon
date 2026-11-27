from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

def load_data(seed=42):
    data=load_breast_cancer()

    X= data.data 
    y= data.target


    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)


    return X_train, X_test, y_train, y_test