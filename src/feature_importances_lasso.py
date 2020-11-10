import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from util import load_data, get_confidence

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', Lasso(alpha=0.005))
])

X, y = load_data()

print('Average precision score  ')
print('-------------')
print('Fold   Score')
k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for i, (train_index, test_index) in enumerate(k_fold.split(X, y)):
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
    y_train, y_test = y[train_index], y[test_index]
    pipeline.fit(X_train, y_train)
    current_coeff = pd.Series(pipeline[1].coef_, X.columns)
    if i == 0:
        coeff = current_coeff
    else:
        coeff = coeff + current_coeff
    
    y_score = get_confidence(pipeline, X_test)
    print(f'{i + 1:<6} {average_precision_score(y_test, y_score):.3f}')
print() 
print(coeff.sort_values(ascending=False))