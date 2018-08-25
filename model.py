import pandas as pd
import numpy as np
import  time

from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import f1_score

import preprocessing as pre


def get_model():
    model = LGBMClassifier(n_estimators=800, subsample=0.3, subsample_freq=1, max_bin=100, num_leaves=15,
                           feature_fraction=0.3, bagging_fraction=0.3, bagging_freq=1,
                           objective='binary', unbalanced=True)
    return model


def load_data(filename=None):
    directory = r'/home/oskar/PycharmProjects/Poverty prediction data'
    if filename is None:
        filename = 'train.csv'
    print("Preprocessing data")
    X, y, ids = pre.main(filenames=[filename], directory=directory, to_binarize=False, to_select_feats=False, to_aggregate=False)
    return X, y, ids


def cv_validate(data=None, scoring='f1_macro'):
    filename = 'train.csv'
    print("Preprocessing data")
    if data is None:
        X, y, ids = load_data()
    else:
        if 'Id' in data.columns:
            X = data.drop(['Target', 'Id', 'idhogar'])
            y = data['Target']
            ids = data['Id']
        else:
            X = data.drop('Target')
            y = data['Target']
            ids = pd.Series([np.nan]*len(X))

    print("Data loaded, validating")
    model = get_model()
    cross_f1 = cross_val_score(model, X, y, scoring=scoring, cv=5)
    return cross_f1


def cv_validate2(data=None, average='macro'):
    filename = 'train.csv'
    print("Preprocessing data")
    if data is None:
        X, y, ids = load_data()
    else:
        if 'Id' in data.columns:
            X = data.drop(['Target', 'Id', 'idhogar'], axis=1)
            y = data['Target']
            ids = data['Id']
        else:
            X = data.drop('Target', axis=1)
            y = data['Target']
            ids = pd.Series([np.nan]*len(X))


    print("Data loaded, validating")
    kf = KFold(n_splits=5)
    cross_f1 = []
    clf = get_model()
    for test_idx, train_idx in kf.split(X):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        print("Training the model for split: {:.2f}% training dataset".format(100*len(X_train)/len(X)))
        clf.fit(X_train, y_train)
        score = f1_score(y_test, clf.predict(X_test), average=average)
        cross_f1.append(score)

    return cross_f1

def validate_dummies(X, y):
    dummies = pd.get_dummies(y)
    scores = []
    for col in dummies.columns:
        data = pd.concat([X, dummies[col]], axis=1)
        data.columns = np.append(X.columns, 'Target')
        score = np.mean(cv_validate2(data, average='binary'))
        scores.append(score)

    return np.mean(scores)



def transform(x, y=None, n=None):
    if n is None:
        if hasattr(x, 'columns'):
            n = len(x.columns)
        else:
            n = len(x[0])
        print("n set to ", n)

    if y is not None:
        print("Starting LDA")
        tr = LDA(n_components=n)
        tr.fit(x, y)
    else:
        print("Starting PCA")
        tr = PCA(n_components=n)
        tr.fit(x)

    x_t = tr.transform(x)
    return x_t


def proba_to_class(value, translator:pd.Series):
    trans_copy = pd.Series(dict(zip(translator.values, translator.index)))
    trans_copy.index = np.cumsum(trans_copy.index)
    for el in trans_copy.index:
        if value < el:
            return trans_copy[el]

def fix_target(row, reference):
    if row['parentesco1'] != 1:
        return reference[row['idhogar']]
    else:
        return row['Target']

def predict(data=None):
    directory = r'/home/oskar/PycharmProjects/Poverty prediction data'
    train_filename = 'train.csv'
    test_filename = 'test.csv'
    filenames = train_filename
    # X, y, ids = pre.main(filenames=filenames)
    # test_idx = y[y.isnull()].index
    # train_idx = y[y.notnull()].index
    # X_train = X.iloc[train_idx]
    # y_train = y.iloc[train_idx].astype(int)
    # X_test = X.iloc[test_idx]
    # test_ids = ids.iloc[test_idx]
    X_train, y_train, ids_train = load_data(train_filename)
    X_test, y_test, ids_test = load_data(test_filename)
    X_test = X_test[X_train.columns.values]

    head_idx = X_train['parentesco1'] == 1
    reference = y_train[head_idx].copy()
    reference.index = X_train[head_idx]['idhogar'].copy()
    y_train = pd.concat([y_train, X_train], axis=1).apply(lambda x: fix_target(x, reference), axis=1)

    selected_features = pre.select_features(X_train, y_train, 100, verbose=1)
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]

    model = get_model()
    print("Training the model")
    model.fit(X_train, y_train)
    print("Predicting values")
    preds_arr = model.predict(X_test)
    preds = pd.DataFrame({'Id' : ids_test,
                       'Target' : preds_arr})
    print("Prediction done")

    # preds_probas_rand = pd.Series(np.random.rand(len(y)))
    # class_probas = y.value_counts(normalize=True)
    # preds_rand = preds_probas_rand.apply(lambda x: proba_to_class(x, class_probas))
    # preds = pd.concat([ids[test_idx], preds_rand[test_idx]], axis=1)
    # preds.columns = ['Id', 'Target']
    # preds['Target'] = preds['Target'].astype(int)
    # print("Prediction done")
    return preds


def export_preds(filename=None):
    preds:pd.DataFrame = predict()
    if filename is None:
        filename = 'predictions.csv'
    preds.to_csv(filename, index=False, header=True)
    print("File saved")


if __name__ == '__main__':
    print("Script started on: ", time.asctime())
    score=cv_validate2()
    print("Average, cross validated f1 score: {:.2f}".format(100*np.mean(score)))
