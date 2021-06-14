from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


class DataPreprocesser():

    def __init__(self):
        self.scaler = None

    def scale_standard(self, data):
        if not self.scaler:
            self.scaler = StandardScaler()
            return self.scaler.fit_transform(data)
        else:
            return self.scaler.transform(data)

    def inverse_scale_standard(self, data):
        return self.scaler.inverse_transform(data)

    def create_sequences(self, x, y, time_steps=30):
        xs, ys = [], []
        for i in range(len(x) - time_steps):
            xs.append(x.iloc[i:(i + time_steps)].values)
            ys.append(y.iloc[i + time_steps])

        return np.array(xs), np.array(ys)

    @staticmethod
    def separate_dataset(x, y, train_percent=80):
        return train_test_split(x, y, train_size=train_percent / 100)
