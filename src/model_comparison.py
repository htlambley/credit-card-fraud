import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, precision_recall_curve
from os import path
from util import load_data, get_confidence, BOOK_PATH

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

X, y = load_data()

k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
print('       Average precision score       ')
print('-------------------------------------')
print('Fold   Classifier               Score')
for i, (train_index, test_index) in enumerate(k_fold.split(X, y)):
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
    y_train, y_test = y[train_index], y[test_index]
    plt.figure()
    for name, classifier in classifiers:
        classifier.fit(X_train, y_train)
        y_score = get_confidence(classifier, X_test)
        print(f'{i + 1:<6} {name:<24} {average_precision_score(y_test, y_score):.3f}')
        curve = precision_recall_curve(y_test, y_score)
        plt.plot(curve[0], curve[1], label=name)
    plt.legend(loc='lower left')
    plt.title(f'Fold {i + 1} precision—recall curve for classification models')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(path.join(BOOK_PATH, 'images', f'final_fold{i + 1}.png'))
    print()

