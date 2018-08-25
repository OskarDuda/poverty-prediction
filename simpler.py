###########
##Imports##
###########

# Basic libraries
import collections
import numpy as np
import pandas as pd
import itertools as it

# Sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder

# System
from os.path import join

def load_data(path, subsampling=None, seed=17):
    if subsampling:
        data = pd.read_csv(path).sample(n=subsampling, random_state=seed)

    else:
        data = raw_data = pd.read_csv(path)

    return data


def binarize(X:pd.DataFrame, columns):
    # binarizer = LabelBinarizer()
    # output = X.copy()
    # for col in columns:
    #     binarizer.fit(X[col])
    #     binned_names = [col + '_' + str(x) for x in binarizer.classes_]
    #     binned_values = np.transpose(binarizer.transform(output[col]))
    #     d = dict(zip(binned_names, binned_values))
    #     output = output.assign(**d)
    #     output.drop(col, inplace=True, axis=1)

    output = X.copy()
    for col in columns:
        binned_values = pd.get_dummies(output[col], drop_first=True)
        binned_names = [col + '_' + str(x) for x in binned_values.columns]
        binned_values.columns = binned_names
        output = pd.concat([output, binned_values], axis=1)
        output.drop(col, inplace=True, axis=1)

    return output


def encode(X):
    encoder = LabelEncoder()
    output = X.copy()
    for col in output.columns:
        encoder.fit(output[col])
        output[col] = encoder.transform(output[col])

    return output


def imput(df, by_cols, to_col, model=None):
    def imput_snippet(X, y, model):
        no_nan_indeces = y.dropna().index
        nan_indeces = y[y.isna()].index

        model.fit(X.iloc[no_nan_indeces],
                y.iloc[no_nan_indeces])

        inputed_values = model.predict(X.iloc[nan_indeces])

        return inputed_values

    output = df.copy().reindex()

    useful_idx = output[by_cols].dropna().index
    to_input = output[to_col].iloc[useful_idx]
    if model is not None:
        clf = model
    else:
        clf = KNeighborsRegressor(n_neighbors=100)

    nan_indeces = output.iloc[useful_idx][output[to_col].iloc[useful_idx].isnull()].index

    output[to_col].iloc[nan_indeces] = imput_snippet(output[by_cols].iloc[useful_idx],
                                        to_input,
                                        clf)

    return output


def age_to_bins(age):
    return age%10


def fix_target(data):
    def has_head(row):
        same_household = data['idhogar'] == row['idhogar']
        is_head = data['parentesco1'] == 1
        return (same_household & is_head).any()

    def get_true_target(row):
        if pd.isnull(row['Target']):
            return np.nan
        else:
            same_household = data['idhogar'] == row['idhogar']
            is_head = data['parentesco1'] == 1
            head = data[same_household & is_head]
            return int(head['Target'])

    df = data.copy()
    df = df[df.apply(has_head, axis=1)]

    not_head = df['parentesco1'] != 1
    df.set_value(index=not_head[not_head].index,
                 col='Target',
                 value=df[not_head].apply(get_true_target, axis=1))
    return df


def numerate_yes_nos(row):
    if row == 'yes':
        return 1
    elif row == 'no':
        return 0
    else:
        return float(row)

def is_iterable(x):
    if isinstance(x, str):
        return False
    elif isinstance(x, collections.Iterable):
        return True


def main(filenames, directory=None, to_binarize=False, to_select_feats=False, to_aggregate=False):
    if is_iterable(filenames):
        filenames = filenames[0]

    if directory is None:
        directory = ''

    full_path = join(directory, filenames)
    data = load_data(full_path)

    data = data.sample(frac=1).reset_index()

    ids = data['Id']
    if 'Target' in data.columns:
        X = data.drop(['Target', 'Id', 'idhogar'], axis=1).copy()
        y = data['Target'].copy()
    else:
        X = data.drop(['Id', 'idhogar'], axis=1).copy()
        y = pd.DataFrame([np.nan]*len(data))

    X.loc[X['meaneduc'].isnull(), 'meaneduc'] = 0
    X.loc[X['SQBmeaned'].isnull(), 'SQBmeaned'] = 0
    X.loc[X['v18q1'].isnull(), 'v18q1'] = X.loc[X['v18q1'].isnull(), 'v18q']

    notnas = X['rez_esc'].notnull()
    nas = X['rez_esc'].isnull()
    clf = KNeighborsClassifier(n_neighbors=20)
    clf.fit(X[notnas][['age', 'agesq', 'instlevel3']], X[notnas]['rez_esc'])
    X.loc[nas, 'rez_esc'] = clf.predict(X[X['rez_esc'].isnull()][['age', 'agesq','instlevel3']])

    notnas = X['v2a1'].notnull()
    nas = X['v2a1'].isnull()
    clf = LinearRegression()
    clf.fit(pd.DataFrame(X.loc[notnas, 'rooms']), X.loc[notnas, 'v2a1'])
    X.loc[nas, 'v2a1'] = clf.predict(pd.DataFrame(X.loc[nas, 'rooms']))

    X['dependency'] = X['dependency'].apply(lambda x: numerate_yes_nos(x))
    X['edjefe'] = X['edjefe'].apply(lambda x: numerate_yes_nos(x))
    X['edjefa'] = X['edjefa'].apply(lambda x: numerate_yes_nos(x))



    return X, y, ids

