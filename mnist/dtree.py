import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import tree


def load(train_fn: str,
         test_fn: str):
    train_df = pd.read_csv(train_fn)
    test_df = pd.read_csv(test_fn)

    X_train = np.array([[int(x) for x in s.split(',')] for s in train_df['Image']])
    y_train = np.array([int(s) for s in train_df['Category']])

    X_test = np.array([[int(x) for x in s.split(',')] for s in test_df['Image']])

    return (X_train, y_train), X_test


def train(X_train: np.array,
          y_train: np.array):
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=1337)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print('Accuracy: {}'.format(acc))

    return clf


def predict(clf,
            X_test: np.array,
            output_fn: str):
    y_pred = clf.predict(X_test)
    test_df = pd.DataFrame(data={'Category': y_pred})
    test_df.to_csv(output_fn, index_label='Id')


if __name__ == '__main__':
    (X_train, y_train), X_test = load(
        train_fn='train.csv',
        test_fn='test.csv')

    model = train(X_train, y_train)
    predict(model, X_test, 'submission.csv')