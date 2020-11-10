from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support
from util import load_data

# To simplify usage in the VotingClassifier, we will use a RidgeClassifier instead of
# a LinearRegression model. This should behave very similarly with the regularisation 
# term so small.
linear_pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
    ('scaler', StandardScaler()),
    ('clf', RidgeClassifier(alpha=0.00001))
])

nn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf',  MLPClassifier(hidden_layer_sizes=(10,), alpha=0.001, random_state=0))
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
    ('Linear Model', linear_pipeline),
    ('Neural Network', nn_pipeline),
    ('KNN', knn_pipeline),
    ('SVM', svm_pipeline),
    ('Random Forest', RandomForestClassifier(n_jobs=-1, random_state=0))
]

voting_clf = VotingClassifier(classifiers)
X, y = load_data()

k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for i, (train_index, test_index) in enumerate(k_fold.split(X, y)):
    # For brevity, we will just test the model on the first fold.
    if i > 0:
        break
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
    y_train, y_test = y[train_index], y[test_index]
    voting_clf.fit(X_train, y_train)
    y_pred = voting_clf.predict(X_test)
    precision, recall, f_score, _ = precision_recall_fscore_support(y_test, y_pred)
    # Precision, recall and f_score are calculated with respect to both classes,
    # but we are only interested in the fraud class (1).
    print(f'Precision: {precision[1]:.3f}')
    print(f'Recall: {recall[1]:.3f}')
    print(f'f-score: {f_score[1]:.3f}')

