import pandas as pd
DATASET_PATH = '../creditcard.csv'
BOOK_PATH = '../mdbook/src'

def load_data():
    data = pd.read_csv(DATASET_PATH)
    y = data.loc[:, 'Class']
    X = data.drop('Class', axis=1)
    return X, y

def get_confidence(clf, X_test):
    if hasattr(clf, 'predict_proba'):
        return clf.predict_proba(X_test)[:, 1]
    elif hasattr(clf, 'decision_function'):
        return clf.decision_function(X_test)
    else:
        return clf.predict(X_test)