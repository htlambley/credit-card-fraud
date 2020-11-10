from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from util import load_data, get_confidence

classifier = SVC()
X, y = load_data()

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', classifier)
])

k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
print('Average precision score  ')
print('-------------')
print('Fold   Score')
for i, (train_index, test_index) in enumerate(k_fold.split(X, y)):
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
    y_train, y_test = y[train_index], y[test_index]
    
    pipeline.fit(X_train, y_train)
    y_score = get_confidence(pipeline, X_test)
    print(f'{i + 1:<6} {average_precision_score(y_test, y_score):.3f}')
    print()

