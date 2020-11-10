from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from eli5 import formatters
from eli5.sklearn import PermutationImportance
from util import load_data

# This helper function can be passed to the PermutationImportance instance
# so the feature importance weights represent the increase/decrease of the 
# average precision score upon permutation of a given feature.
def score(clf, X, y):
    y_score = clf.predict_proba(X)[:, 1]
    return average_precision_score(y, y_score)

classifier = RandomForestClassifier(n_jobs=-1, random_state=0)

X, y = load_data()
# Use only the first 10,000 samples to speed up the process. Calculating
# the feature importances takes a large amount of time because we 
# must permute each feature and retrain.
X = X[:10000]
y = y[:10000]

k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for i, (train_index, test_index) in enumerate(k_fold.split(X, y)):
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
    y_train, y_test = y[train_index], y[test_index]
    classifier.fit(X_train, y_train)
    perm = PermutationImportance(classifier, scoring=score).fit(X_test, y_test)
    # We will create an aggregate feature importance ranking by storing the weights from each fold
    # as a dataframe and summing them all together.
    if i == 0:
        explanations = formatters.explain_weights_df(perm, feature_names=X_train.columns.to_numpy())
        explanations.set_index('feature', inplace=True)
    else:
        explanation = formatters.explain_weights_df(perm, feature_names=X_train.columns.to_numpy())
        explanation.set_index('feature', inplace=True)
        explanations = explanations + explanation
# Output the aggregate feature importances after sorting to show the highest ranked first.
print(explanations.weight.sort_values(ascending=False))
