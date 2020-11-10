import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, precision_recall_curve
from os import path
from util import load_data, get_confidence, BOOK_PATH

classifiers = [
    ('Linear Regression', LinearRegression()),
    ('Gaussian Naive Bayes', GaussianNB()),
    ('Support Vector Machine', LinearSVC(dual=False)),
    ('K Nearest Neighbours', KNeighborsClassifier()),
    ('Neural Network', MLPClassifier(random_state=0)),
    ('Random Forest', RandomForestClassifier(n_jobs=-1, random_state=0))
]

# load_data() is a helper to load the dataset into memory and partition into
# features and targets. We will use this throughout and the full code is available
# in the source folder.
X, y = load_data()

# We use a stratified fold to ensure that the class balance is preserved. 
# Otherwise we could have the validation data having a greater or smaller
# number of fraudulent cases, which would affect the generalisation score.
k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

print('       Average precision score       ')
print('-------------------------------------')
print('Fold   Classifier               Score')
for i, (train_index, test_index) in enumerate(k_fold.split(X, y)):
    # Use the indices given by the StratifiedKFold to generate train and 
    # test sets.
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
    y_train, y_test = y[train_index], y[test_index]

    # Begin a new figure which will hold the precision–recall curve, then iterate
    # through all classifiers to fit and test performance.
    plt.figure()
    for name, classifier in classifiers:
        classifier.fit(X_train, y_train)
        # get_confidence is another helper function. As some models can output scores using 
        # predict_proba, while others only support decision_function (or in the case of using
        # a regression model to estimate a score, predict), this function determines the 
        # appropriate method of obtaining some numerical score from the model. 
        #
        # We will use get_confidence throughout the rest of this section.
        y_score = get_confidence(classifier, X_test)
        print(f'{i + 1:<6} {name:<24} {average_precision_score(y_test, y_score):.3f}')
        # The precision_recall_curve will test the performance of the model at different
        # thresholds.
        curve = precision_recall_curve(y_test, y_score)
        plt.plot(curve[0], curve[1], label=name)
    # Label and save precision–recall plot.
    plt.legend(loc='lower left')
    plt.title(f'Fold {i + 1} precision—recall curve for baseline classification models')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(path.join(BOOK_PATH, 'images', f'fold{i + 1}.png'))
    print()

