from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class SBS:

    def __init__(self, estimator, k_features, scoring=accuracy_score, test_size=0.25, random_state=1):
        self.scoring = scoring  # 特徴量の評価指標
        self.estimator = estimator  # 推定器
        self.k_features = k_features  # 選択する特徴量の個数
        self.test_size = test_size  # テストデータの割合
        self.random_state = random_state  # 乱数シード
        self.indices_ = tuple()  # 現在の次元
        self.subsets_ = []  # 次元削減の記録
        self.scores_ = []  # スコアの記録
        self.k_score_ = 0  # 最終的なスコア

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=self.test_size, random_state=self.random_state)
        dim = X_train.shape[1]  # 全ての特徴量の個数
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train,
                                 X_test, y_test, self.indices_)
        self.scores_ = [score]
        while dim > self.k_features:
            scores = []
            subsets = []
            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train, y_train, X_test, y_test, p)
                scores.append(score)
                subsets.append(p)
            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1
            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score