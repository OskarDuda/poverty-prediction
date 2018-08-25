import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import join

from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

directory = r"C:\Users\duos8001\Documents\Python\Machine Learning tutorial\Poverty prediction"
filename = "train.csv"

df = pd.read_csv(join(directory, filename))

df = df.dropna(axis=1)

acc_dtypes = [np.int64, np.float64]
is_numeric = df.dtypes.apply(lambda x: x in acc_dtypes)
df = df[is_numeric.index[is_numeric]]


# model = RandomForestClassifier(n_estimators=6500, verbose=2, max_features=8)
df = df.sample(frac=1, random_state=17)
X = df.drop('Target', axis=1)
# useful_cols = ['rooms', 'r4h1', 'refrig', 'overcrowding', 'r4t3', 'r4t2', 'r4h2',
#        'r4m3', 'tamviv', 'noelec', 'instlevel9', 'paredblolad', 'planpri',
#        'paredpreb', 'energcocinar3', 'paredmad', 'energcocinar4', 'pareddes',
#        'elimbasu4', 'abastaguafuera', 'r4t1', 'v14a', 'female', 'r4m2',
#        'parentesco9', 'elimbasu3', 'paredzinc', 'r4h3', 'techocane', 'hhsize',
#        'paredother', 'sanitario6', 'sanitario5', 'abastaguano', 'paredfibras',
#        'energcocinar1', 'instlevel6', 'sanitario2', 'estadocivil4']
# X = X[useful_cols]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, shuffle = False, stratify=None)
model = GradientBoostingClassifier(n_estimators=1000, subsample=0.8, max_features=8, verbose=1)
print("Training model")
model.fit(X_train, y_train)
print("Validating")
accuracy = model.score(X_test, y_test)
# f1 = f1_score(y_test, model.predict(X_test), average = 'macro')
f1 = f1_score(y_test, list(model.predict(X_test)), average = 'macro')

cross_f1 = cross_val_score(model, X, y, scoring='f1_macro', cv=5)
print("Prediction accuracy: {:.2f}%".format(accuracy*100))
print("F1 macro score on test set: {:.2f}%".format(100*f1))
print("Cross validated f1 score: {:.2f}%".format(100*np.mean(cross_f1)))