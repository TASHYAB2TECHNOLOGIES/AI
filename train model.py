import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
from features import features_from_window

def load_dataset(path='data/dataset.npz'):
    data = np.load(path, allow_pickle=True)
    return data['X'], data['y'], int(data['fs'])

def build_feature_matrix(X, fs):
    feats = []
    for w in X:
        vec, keys = features_from_window(w, fs=fs)
        feats.append(vec)
    return np.vstack(feats), keys

def train_and_save(path='data/dataset.npz', out_dir='models'):
    Xraw, y, fs = load_dataset(path)
    Xfeat, keys = build_feature_matrix(Xraw, fs)

    os.makedirs(out_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        Xfeat, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training RandomForest...")
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    ypred_rf = rf.predict(X_test)
    print(classification_report(y_test, ypred_rf))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, ypred_rf))
    joblib.dump((rf, keys), os.path.join(out_dir, 'rf_model.joblib'))

    print("Training SVM...")
    svm = SVC(kernel='rbf', C=1.0, probability=True)
    svm.fit(X_train, y_train)
    ypred_svm = svm.predict(X_test)
    print(classification_report(y_test, ypred_svm))
    joblib.dump((svm, keys), os.path.join(out_dir, 'svm_model.joblib'))

if __name__ == "__main__":
    train_and_save()
