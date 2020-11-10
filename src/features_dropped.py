import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, precision_recall_curve
from os import path
from util import load_data, get_confidence, BOOK_PATH

#features_to_keep = ['V3', 'V4', 'V9', 'V16', 'V10', 'V11', 'V12', 'V14', 'V17']
classifier = LGBMClassifier(boosting_type='goss')

X, y = load_data()
#X = X.loc[:, features_to_keep]

k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
print('Average precision score  ')
print('-------------')
print('Fold   Score')
for i, (train_index, test_index) in enumerate(k_fold.split(X, y)):
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
    y_train, y_test = y[train_index], y[test_index]
    plt.figure()
    classifier.fit(X_train, y_train)
    y_score = get_confidence(classifier, X_test)
    print(f'{i + 1:<6} {average_precision_score(y_test, y_score):.3f}')
    curve = precision_recall_curve(y_test, y_score)
    plt.plot(curve[0], curve[1])
    plt.title(f'Fold {i + 1} precision—recall curve for baseline classification models')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(path.join(BOOK_PATH, 'images', f'dropped_features_fold{i + 1}.png'))
    print()

