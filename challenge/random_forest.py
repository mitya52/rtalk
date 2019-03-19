import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn import ensemble
from sklearn.model_selection import train_test_split


def load(filename: str,
         is_train: bool):
    print('Load data...')
    df = pd.read_csv(filename)
    _X = np.array([[int(x) for x in s.split(',')] for s in df['Image']])
    if is_train:
        _y = np.array([str(s) for s in df['Category']])
    else:
        _y = None
    return _X, _y


def train(X_train: np.array,
          y_train: np.array):
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=1337)

    clf = ensemble.RandomForestClassifier(n_estimators=30,
                                          n_jobs=-1,
                                          verbose=1)
    clf = clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print('Accuracy: {}'.format(acc))

    return clf


def predict(clf,
            X_test: np.array,
            output_fn: str):
    print('Prediction...')
    y_pred = clf.predict(X_test)
    test_df = pd.DataFrame(data={'Category': y_pred})
    test_df.to_csv(output_fn, index_label='Id')


if __name__ == '__main__':
    X_train, y_train = load(
        filename='../../challenge/train.csv',
        is_train=True)

    X_train, y_train = X_train[:1000], y_train[:1000]
    model = train(X_train, y_train)

    del X_train
    del y_train

    X_test, _ = load(
        filename='../../challenge/test.csv',
        is_train=False)
    predict(model, X_test, 'submission.csv')
