import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve, accuracy_score
from util import load_data, get_confidence
import itertools

# We create popelines from the various successful models we've developed.
linear_pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
    ('scaler', StandardScaler()),
    ('clf', LinearRegression())
])

nn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf',  MLPClassifier(hidden_layer_sizes=(10,), alpha=0.001, random_state=2))
])

knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', KNeighborsClassifier())
])

svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC())
])

classifiers = [
    ('Linear Regression', linear_pipeline),
    ('Neural Network', nn_pipeline),
    ('KNN', knn_pipeline),
    ('SVM', svm_pipeline),
    ('Random Forest', RandomForestClassifier(n_jobs=-1, random_state=0))
]


def find_best_threshold(y_true, y_score):
    curve = precision_recall_curve(y_true, y_score)
    thresholds = curve[2]
    accuracy = []
    for threshold in thresholds:
        accuracy.append(accuracy_score(y_true, y_score > threshold))
    return thresholds[np.argmax(accuracy)]

X, y = load_data()

# We will collect the indices of the misclassified transactions (in fold 1) in
# misclassified. We then plot a histogram which shows us how many times each transaction
# was misclassified. We test using 5 classification pipelines, so if a transaction has
# been misclassified by all 5, it will show as a bar of 5.
misclassified = []
k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for i, (train_index, test_index) in enumerate(k_fold.split(X, y)):
    # Only evaluate the first fold for brevity.
    if i > 0:
        break
    
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
    y_train, y_test = y[train_index], y[test_index]
    for name, clf in classifiers:
        clf.fit(X_train, y_train)
        y_score = get_confidence(clf, X_test)
        # We want to test each classifier on its most accurate threshold.
        threshold = find_best_threshold(y_test, y_score)
        # Creates a numpy array that is 1 if the corresponding transaction
        # was correctly classified by clf.
        classified_correctly = (y_score > threshold) == y_test

        # Find the indices of the misclassifications and append to the collection
        # misclassified.
        misclassifications = list(np.where(classified_correctly == 0)[0])
        misclassified.append(misclassifications)
        
    # misclassified contains 5 lists: one from each classifier. We flatten this into a
    # 1 dimensional list.
    misclassified_transactions = list(itertools.chain.from_iterable(misclassified))
    # Plot a histogram of the data to show the number of misclassifications.
    plt.hist(misclassified_transactions, bins=len(misclassified_transactions))
    plt.title('Number of times transaction misclassified by index')
    plt.xlabel('Index')
    plt.ylabel('Number of times misclassified')
    plt.ylim([0, 5])
    plt.show()

