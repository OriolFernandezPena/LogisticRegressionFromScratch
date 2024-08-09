from src.model import LogisticRegression
from src.visualization import plot_ROC
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

    custom_clf_roc_curve = plot_ROC(y_pred, y_iris, title="Logistic Regression from scratch ROC Curve")
    custom_clf_roc_curve.savefig('CustomLogReg_ROC.jpg', format='jpg', dpi=300)

    sk_log = SKLogistic()
    sk_log.fit(X_iris, y_iris)
    y_pred = sk_log.predict_proba(X_iris)[:, 1]
    print(f"AUC Score {roc_auc_score(y_iris, y_pred):.6f}")

    sk_log_roc_curve = plot_ROC(y_pred, y_iris, title="Scikit-Learn Logistic Regression ROC Curve")
    sk_log_roc_curve.savefig('SKLearnLogReg_ROC.jpg', format='jpg', dpi=300)


if __name__ == '__main__':
    main()