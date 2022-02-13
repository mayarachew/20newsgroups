"""Classification."""
import pandas as pd  # type: ignore
from typing import Any
from scipy.sparse import csr_matrix

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV
from .plot_confusion_matrix import plot_confusion_matrix


# def test_hyperparameters(classifier, X_train, y_train):
#     """Function to test hiper-parameters.

#     Args:
#         classifier (Any): classifier
#         X_train (csr_matrix): test sparse matrix
#         y_train (Any): test labels

#     Returns:
#         pd.DataFrame: preprocessed dataFrame
#     """
#     parameters = {}
#     classification = None

#     if classifier == 'Random Forest':
#         parameters = {'n_estimators': [100, 200, 300, 400, 500], "max_depth": [
#             3, 4, 5, 6, 7, 8, None], 'criterion': ['gini', 'entropy'], 'max_features': ['auto', 'sqrt', 'log2']}
#         classification = RandomForestClassifier(random_state=0)
#     elif classifier == 'Naive Bayes':
#         parameters = {'alpha': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]}
#         classification = MultinomialNB()
#     elif classifier == 'SVM':
#         parameters = {'kernel': ['rbf', 'poly', 'sigmoid', 'linear'], "gamma": [
#             1e-3, 1e-4], "C": [1, 10, 100, 1000]}
#         classification = SVC(random_state=0)

#     score = 'f1'

#     print("# Metric: %s" % score)
#     print()

#     clf = GridSearchCV(classification, parameters,
#                        scoring="%s_macro" % score, cv=5)
#     clf.fit(X_train, y_train)

#     print("Best hyper-parameters:")
#     print()
#     print(clf.best_params_)


def create_classifier(classifier: Any, x_train: csr_matrix, y_train: pd.Series, x_test: csr_matrix, y_test: pd.Series) -> pd.Series:
    """Create classification model.

    Args:
        classifier (Any): classifier
        X_train (csr_matrix): train sparse matrix
        y_train (Series): train labels
        x_test (csr_matrix): test sparse matrix
        y_test (pd.Series): test labels

    Returns:
        y_pred (pd.Series): Rótulos dos dados de teste gerados pelo classifier
    """

    # Create classifier
    classifier.fit(x_train, y_train)

    # Define labels
    y_true, y_pred = y_test, classifier.predict(x_test)

    print('Classification report: ')
    print(classification_report(y_true, y_pred, zero_division=1))

    print('Confusion matrix: ')
    plot_confusion_matrix(classifier, x_test, y_test)

    return y_pred
