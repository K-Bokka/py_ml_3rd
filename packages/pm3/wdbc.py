import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class WDBC:

    def __init__(self):
        df = pd.read_csv('../../data/wdbc/wdbc.data', header=None)
        X = df.iloc[:, 2:].values
        y = df.iloc[:, 1].values

        le = LabelEncoder()
        y = le.fit_transform(y)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.20, random_state=1, stratify=y)

    def train_datasets(self):
        return self.X_train, self.y_train

    def test_datasets(self):
        return self.X_test, self.y_test
