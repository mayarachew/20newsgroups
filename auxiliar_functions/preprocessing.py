"""Preprocessing."""
import pandas as pd  # type: ignore
# from typing import Union
import string
from scipy.sparse import csr_matrix

import re
import string
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

# from imblearn.over_sampling import SMOTE

from sklearn.feature_extraction.text import TfidfVectorizer


def preprocessing(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Função de pré-processamento dos textos.

    Args:
        raw_data (pd.DataFrame): raw DataFrame

    Returns:
        pd.DataFrame: preprocessed DataFrame
    """

    # Convert to lower case
    raw_data = [text.lower() for text in raw_data.data]

    # Strip all punctuation
    table = str.maketrans('', '', string.punctuation)
    raw_data = [text.translate(table) for text in raw_data]

    # Convert all numbers to the word 'num'
    raw_data = [re.sub(r'\d+', 'num', text) for text in raw_data]

    # Convert text to tokens
    nltk.download('punkt')
    raw_data = [nltk.word_tokenize(text) for text in raw_data]

    # Remove stopwords
    nltk.download('stopwords')
    stop_words = set(stopwords.words("english"))
    raw_data = [[word for word in text if word not in stop_words]
                for text in raw_data]

    # Stemming
    raw_data = [[LancasterStemmer().stem(word) for word in text]
                for text in raw_data]

    # Lemmatizing
    nltk.download('wordnet')
    raw_data = [[WordNetLemmatizer().lemmatize(word, 'n')
                 for word in text] for text in raw_data]

    # Convert token back to text
    data = [TreebankWordDetokenizer().detokenize(text).split()
            for text in raw_data]
    data_tfidf = [TreebankWordDetokenizer().detokenize(text)
                  for text in raw_data]

    return data, data_tfidf


def convert_to_matrix_tfidf(data: pd.DataFrame, X_train: pd.DataFrame, X_test: pd.DataFrame) -> csr_matrix:
    """Function to convert data into a tfidf matrix.

    Args:
        dados (pd.DataFrame): DataFrame
        X_train (pd.DataFrame): train DataFrame
        X_test (pd.DataFrame): test DataFrame

    Returns:
        tf_train (csr_matrix): tf-idf train sparse matrix
        tf_test (csr_matrix): tf-idf test sparse matrix
        tfidf (csr_matrix): tf-idf sparse matrix
    """
    # Convert data into a matrix of TF-IDF features
    vectorizer = TfidfVectorizer(
        sublinear_tf=True, stop_words='english', max_df=0.95, min_df=4)
    vectorizer.fit(X_train)

    # Without dimensionality reduction
    # For classification
    tf_train = vectorizer.transform(X_train)
    tf_test = vectorizer.transform(X_test)

    # For visualization
    tfidf = vectorizer.fit_transform(data)
    print(tfidf.shape)

    return tf_train, tf_test, tfidf


# def over_sampling(dados: pd.DataFrame, tfidf: csr_matrix, attribute: str) -> Union[csr_matrix, pd.Series, csr_matrix, pd.Series]:
#     """Function to apply over_sampling with SMOTE.

#     Args:
#         dados (pd.DataFrame): dataFrame
#         tfidf (csr_matrix): tf-idf sparse matrix
#         attribute (string): column to be balanced

#     Returns:
#         X_train (csr_matrix): tf-idf train sparse matrix
#         y_train (pd.Series): train labels
#         X_test (csr_matrix): tf-idf test sparse matrix
#         y_test (pd.Series): test labels
#     """
#     oversample = SMOTE()
#     X_train, y_train = oversample.fit_resample(
#         tfidf[:5713], dados[:5713][attribute])

#     X_test = tfidf[5714:]
#     y_test = dados[5714:][attribute]

#     return X_train, y_train, X_test, y_test
