from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy


class DataPreprocesser():

    def __init__(self, dataset):
        self.dataset = dataset

    def scale_standard(self):
        min_max_scaler = preprocessing.StandardScaler()
        date_column = self.dataset.select_dtypes(exclude=['float64'])
        date_column_name = date_column.columns.values[0]
        self.dataset['minutes'] = date_column[date_column_name].dt.minute
        self.dataset['seconds'] = date_column[date_column_name].dt.second
        print(self.dataset)
        df = self.dataset.drop([date_column_name], axis=1).values
        print(df)
        np_scaled = min_max_scaler.fit_transform(df)
        scaled_dataframe = pd.DataFrame(np_scaled)
        scaled_dataframe = scaled_dataframe.dropna()
        # print(scaled_dataframe)
        return scaled_dataframe

    def scale_minmax(self):
        self.scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        scaled_dataset = self.scaler.fit_transform(self.dataset)
        return scaled_dataset

    def invert_scale_minmax(self, data):
        return self.scaler.inverse_transform(data)

    @staticmethod
    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        # print(numpy.array(dataX))
        # print(numpy.array(dataY))
        return numpy.array(dataX), numpy.array(dataY)

    @staticmethod
    def separate_dataset(x, y, train_percent=80):
        return train_test_split(x, y, train_size=train_percent / 100)
