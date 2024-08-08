from src.model import LogisticRegression
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression as SKLogistic

def main():
    iris = datasets.load_iris()

    X_iris = iris.data
    y_iris = (iris.target == 2).astype(int)

    custom_clf = LogisticRegression()
    custom_clf.fit(X_iris, y_iris)
    print(custom_clf.theta)

    sk_log = SKLogistic()
    sk_log.fit(X_iris, y_iris)
    print(sk_log.coef_)


if __name__ == '__main__':
    main()