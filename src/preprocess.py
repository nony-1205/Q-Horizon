from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def preprocess(X_train,X_test,n_components):

    scaler = StandardScaler()

    X_train_scale= scaler.fit_transform(X_train)
    X_test_scale= scaler.transform(X_test)

    pca = PCA(n_components=n_components)
    X_train_pca=pca.fit_transform(X_train_scale)
    X_test_pca= pca.transform(X_test_scale)

    return X_train_scale,X_test_scale, X_train_pca, X_test_pca, pca

