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

#
# def input(X, y, model=None):
#     if model:
#         clf = model
#     else:
#         clf = KNeighborsRegressor(n_neighbors=100)
#
#     no_nan_indeces = y.dropna().index
#     nan_indeces = y[y.isna()].index
#
#     clf.fit(X.loc[no_nan_indeces],
#             y[no_nan_indeces])
#
#     inputed_values = clf.predict(X.loc[nan_indeces])
#
#     return inputed_values
#
#
# def full_input_by(data, by_cols, to_col, model=None):
#     def input(X, y, model):
#         no_nan_indeces = y.dropna().index
#         nan_indeces = y[y.isna()].index
#
#         model.fit(X.loc[no_nan_indeces],
#                   y[no_nan_indeces])
#
#         inputed_values = model.predict(X.loc[nan_indeces])
#
#         return inputed_values
#
#     output = data.copy()
#
#     useful_idx = output[by_cols].dropna().index
#     to_input = output[to_col][useful_idx]
#     if model:
#         clf = model
#     else:
#         clf = KNeighborsRegressor(n_neighbors=100)
#
#     nan_indeces = output.loc[useful_idx][output[to_col][useful_idx].isnull()].index
#
#     output[to_col][nan_indeces] = input(output[by_cols].loc[useful_idx],
#                                         to_input,
#                                         clf)
#
#     return output


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


def select_features(X, y, n=None, verbose=0, step=2):
    selector = RFE(RandomForestClassifier(n_estimators=150), n_features_to_select=n, verbose=verbose, step=step)
    selector.fit(X, y)

    return X.columns[selector.support_]


def create_aggregate_features(X):
    agg_feats = ['min', 'max', 'mean', 'std', 'sum']
    agg_df = pd.DataFrame([])

    for agg_feat in agg_feats:
        basic_to_agg_names = {x: x + '_' + agg_feat for x in X.columns}
        agg_tmp = X.groupby('idhogar').agg(agg_feat).rename(columns=basic_to_agg_names)
        agg_df = pd.concat([agg_df, agg_tmp], axis=1)

    uniques_per_col = [[col, len(agg_df[col].value_counts())] for col in agg_df.columns]
    to_drop = [x[0] for x in uniques_per_col if x[1] < 2]
    agg_df.drop(columns=to_drop, inplace=True)
    agg_nulls = agg_df.isnull().any(axis=0)
    agg_nulls = agg_nulls[agg_nulls].index
    for col in agg_nulls:
        agg_df[col] = agg_df[col].value_counts().argmax()

    return agg_df


def main(directory: str=None, filenames: str=None, to_binarize=True, to_numerate=True,
         to_imput=True, to_select_feats=True, to_aggregate=True):
    """Takes directory and filename of the training data. Return preprocessed data separated into X and y pd.Dataframes
    :str directory:
    :str filename:
    :bool to_binarize:
    :bool to_numerate:
    :bool to_imput:
    :bool to_select_feats:
    :return X:pd.Dataframe, y:pd.Dataframe:
    """
    if directory is None:
        directory = ''
    if not is_iterable(filenames):
        names = [filenames]
    else:
        names = filenames

    # Read the csv file
    # if is_iterable(filenames):
    #     data = pd.DataFrame([])
    data = pd.DataFrame([])
    for name in names:
        tmp = pd.read_csv(name)
        if 'Target' not in tmp.columns:
            tmp['Target'] = np.nan
        data = pd.concat([tmp, data])
    # else:
    #     data = pd.read_csv(join(directory, filenames))
    #     if 'Target' not in data.columns:
    #         data['Target'] = np.nan


    # Shuffle the data and reset the index
    data = fix_target(data)
    data = data.sample(frac=1, random_state=17).reset_index()
    print("Data shuffled")


    # Split the data into dependent and independent vars and ids
    X: pd.DataFrame = data.drop(['Target', 'Id'], axis=1).copy()
    y: pd.Series = data['Target'].copy()
    ids: pd.Series = data['Id']
    print("Data split")


    # Convert data to numeric, where possible
    if to_numerate is True:
        le = LabelEncoder()
        le.fit(X['idhogar'])

        X.loc[:, 'idhogar'] = le.transform(X['idhogar'])
        X.loc[:, 'edjefa'] = X['edjefa'].apply(numerate_yes_nos)
        X.loc[:, 'edjefe'] = X['edjefe'].apply(numerate_yes_nos)
        X.loc[:, 'dependency'] = X['dependency'].apply(numerate_yes_nos)
        X.loc[:, 'age'] = X['age'].apply(age_to_bins)

        print("Data converted to numeric")


    # Develop features aggregated across households
    if to_aggregate is True:
        agg_df = create_aggregate_features(X)
        X = pd.merge(X, agg_df, on='idhogar')
        print("Created {} new aggregated features".format(len(agg_df.columns)))


    # Imput nan values
    if to_imput is True:
        X.update(imput(X, ['SQBescolari'], 'meaneduc'), overwrite=True)
        X.update(imput(X, ['meaneduc'], 'SQBmeaned'), overwrite=True)
        X.update(imput(X, ['rooms', 'meaneduc','SQBmeaned', 'SQBedjefe'], 'v2a1'), overwrite=True)
        X.update(imput(X, ['agesq', 'SQBage','age'], 'rez_esc', model=KNeighborsClassifier(n_neighbors=40)), overwrite=True)
        X.loc[X.v18q1.isnull(), 'v18q1'] = 0

        print("Nan values imputed")


    # Drop non numeric and redundant columns
    X = X.dropna(axis=1)
    acc_dtypes = [np.int64, np.float64]
    is_numeric = X.dtypes.apply(lambda x: x in acc_dtypes)
    X = X[is_numeric.index[is_numeric]]
    X.drop('index', inplace=True, axis=1)
    print("Redundant columns dropped. New number of features is {}".format(len(X.columns)))


    # Binarize the features
    if to_binarize is True:
        cols_to_binarize = [x for x in X.columns if len(X[x].value_counts()) < 10]
        X = binarize(X, cols_to_binarize)

        print("Features binarized. New number of features is {}".format(len(X.columns)))


    # Feature selection
    if to_select_feats is True:
        selected_columns = select_features(X[y.notnull()], y[y.notnull()], n=400, verbose=1, step=10)
        X = X[selected_columns]

        print("Selected {} features".format(len(X.columns)))


    return X, y, ids
