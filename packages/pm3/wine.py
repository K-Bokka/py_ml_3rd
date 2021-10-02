from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


class Wine:

    def __init__(self):
        self.df = pd.read_csv('../../data/wine/wine.data', header=None)
        self.df.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                           'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids',
                           'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue',
                           'OD280/OD315 of diluted wines', 'Proline']

        X = self.df.iloc[:, 1:].values
        y = self.df.iloc[:, 0].values

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=0, stratify=y)

        sc = StandardScaler()
        sc.fit(self.X_train)
        self.X_train_std = sc.transform(self.X_train)
        self.X_test_std = sc.transform(self.X_test)

    def all_datasets(self):
        return self.df

    def train_datasets(self):
        return [self.X_train, self.y_train]

    def test_datasets(self):
        return [self.X_test, self.y_test]

    def train_std_datasets(self):
        return [self.X_train_std, self.y_train]

    def test_std_datasets(self):
        return [self.X_test_std, self.y_test]

    def combined_std_datasets(self):
        return [np.vstack((self.X_train_std, self.X_test_std)), np.hstack((self.y_train, self.y_test))]

    def combined_datasets(self):
        return [np.vstack((self.X_train, self.X_test)), np.hstack((self.y_train, self.y_test))]
