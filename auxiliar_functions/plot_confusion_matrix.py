"""Confusion matrix."""
import matplotlib.pyplot as plt  # type: ignore
from sklearn.metrics import ConfusionMatrixDisplay


def plot_confusion_matrix(classifier, x_test, y_test):
    """Function to plot a confusion matrix.

    Args:
        classifier (Any): Trained model
        x_test (csr_matrix): Test sparse matrix
        y_test (Series): Test labels

    Returns:
        None
    """
    ConfusionMatrixDisplay.from_estimator(
        classifier, x_test, y_test, cmap=plt.cm.Blues)
    plt.show()

    return
