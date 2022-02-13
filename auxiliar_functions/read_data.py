"""Read data."""
from sklearn.datasets import fetch_20newsgroups


def read_data():
    """Function to plot a confusion matrix.

    Args:
        classifier (Any): trained model
        x_test (csr_matrix): test sparse matrix
        y_test (Series): test labels

    Returns:
        None
    """
    # Define 7 news groups
    selected_groups = ['alt.atheism', 'comp.sys.mac.hardware', 'comp.windows.x',
                       'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med']

    # Read 20 news groups folders of the 20newsgroups dataset
    newsgroups = fetch_20newsgroups(
        subset='train', categories=selected_groups, remove=('headers', 'footers', 'quotes'))

    return newsgroups
