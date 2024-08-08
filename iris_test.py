from src.model import LogisticRegression
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression as SKLogistic
from sklearn.metrics import roc_auc_score

def main():
    iris = datasets.load_iris()

    X_iris = iris.data
    y_iris = (iris.target == 2).astype(int)

    custom_clf = LogisticRegression()
    custom_clf.fit(X_iris, y_iris)
    y_pred = custom_clf.predict_proba(X_iris)
    print(f"AUC Score {roc_auc_score(y_iris, y_pred):.6f}")

    sk_log = SKLogistic()
    sk_log.fit(X_iris, y_iris)
    y_pred = sk_log.predict_proba(X_iris)[:, 1]
    print(f"AUC Score {roc_auc_score(y_iris, y_pred):.6f}")


if __name__ == '__main__':
    main()