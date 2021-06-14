from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Dropout
from keras.models import Sequential
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


class LSTM_Autoencoder:
    def __init__(self, optimizer='adam', loss='mse'):
        self.optimizer = optimizer
        self.loss = loss
        self.n_features = 1

    def build_model(self):
        timesteps = self.timesteps
        n_features = self.n_features
        model = Sequential()

        # Encoder
        model.add(LSTM(128, activation='relu', input_shape=(timesteps, n_features)))
        # model.add(LSTM(100, activation='relu', return_sequences=True))
        # model.add(LSTM(1, activation='relu'))
        model.add(Dropout(rate=0.2))
        model.add(RepeatVector(timesteps))

        # Decoder
        model.add(LSTM(128, activation='relu', return_sequences=True))
        # model.add(LSTM(16, activation='relu', return_sequences=True))
        model.add(Dropout(rate=0.2))
        model.add(TimeDistributed(Dense(n_features)))

        model.compile(optimizer=self.optimizer, loss=self.loss)
        model.summary()
        self.model = model

    def fit(self, X, Y, epochs=3, batch_size=32):
        self.timesteps = X.shape[1]
        self.build_model()

        input_X = np.expand_dims(X, axis=2)
        # input_Y = np.expand_dims(Y, axis=2)
        # input_X = X
        self.model.fit(input_X, Y, epochs=epochs, batch_size=batch_size)

    def predict(self, X):
        input_X = np.expand_dims(X, axis=2)
        # input_X = X
        output_X = self.model.predict(input_X)
        # reconstruction = np.squeeze(output_X)
        # return reconstruction
        reconstruction = np.squeeze(output_X)
        return np.mean(np.abs(reconstruction - X), axis=1)
        # return np.linalg.norm(X - reconstruction, axis=-1)

    def plot(self, scores, timeseries, threshold=0.95):
        sorted_scores = sorted(scores)
        threshold_score = sorted_scores[round(len(scores) * threshold)]

        anomalous = np.where(scores > threshold_score)
        normal = np.where(scores <= threshold_score)

        plt.figure(figsize=(25, 3))
        plt.title("Anomalies")
        plt.scatter(normal, timeseries[normal][:, -1], s=3)
        plt.scatter(anomalous, timeseries[anomalous][:, -1], s=5, c='r')
        plt.show()


class LSTM_RNN:
    def __init__(self):
        self.timesteps = 2
        self.n_features = 1

    def build_model(self):
        model = Sequential()
        model.add(LSTM(16, activation='relu', input_shape=(self.timesteps, self.n_features), return_sequences=True))
        model.add(LSTM(16, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.summary()
        self.model = model

    def fit(self, X_train, Y_train):
        self.build_model()
        self.model.fit(X_train, Y_train, epochs=30, batch_size=32)

    def predict(self, X_test):
        return self.model.predict(X_test)


from anomalydetection.data_preprocessing import DataPreprocesser

train_dataframe = pd.read_csv('../train_dataset.csv', usecols=[0], engine='python')
test_dataframe = pd.read_csv('../test_dataset_outliers.csv', usecols=[0], engine='python')
train_dataset = train_dataframe.values
test_dataset = test_dataframe.values

processer = DataPreprocesser(train_dataset)
trainX, trainY = processer.create_dataset(train_dataset, 1)
testX, testY = processer.create_dataset(test_dataset, 1)
# trainX = trainX.reshape(len(trainX), 3, 1)
# testX = testX.reshape(len(testX), 3, 1)

lstm_autoencoder = LSTM_Autoencoder(optimizer='adam', loss='mae')
lstm_autoencoder.fit(trainX, trainY, epochs=100, batch_size=32)
scores = lstm_autoencoder.predict(testX)
lstm_autoencoder.plot(scores, testX, threshold=0.90)

predY = lstm_autoencoder.predict(trainX)
lstm_autoencoder.plot(predY, trainX, threshold=0.90)
# plt.figure(figsize=(25, 3))
# plt.plot(testY)
# plt.plot(predY)
# plt.show()

# processer = DataPreprocesser(train_dataset)
# trainX, trainY = processer.create_dataset(train_dataset, 2)
# testX, testY = processer.create_dataset(test_dataset, 2)
# trainX = trainX.reshape(len(trainX), 2, 1)
# trainY = trainY.reshape(len(trainY), 1)
# testX = testX.reshape(len(testX), 2, 1)
# lstm_rnn = LSTM_RNN()
# lstm_rnn.fit(trainX, trainY)
# predY = lstm_rnn.predict(testX)
from sklearn.metrics import mean_squared_error
# print("MSE:", mean_squared_error(testY, predY))

# plt.figure(figsize=(25, 3))
# plt.plot(testY)
# plt.plot(predY)
# plt.show()

# trainX = trainX.reshape((trainX.shape[0], trainX.shape[1], 1))
# trainY = trainY.reshape((trainY.shape[0], 1, 1))
#
# model = Sequential()
# model.add(LSTM(100, activation='relu', input_shape=(3, 1)))
# model.add(RepeatVector(1))
# model.add(LSTM(100, activation='relu', return_sequences=True))
# model.add(TimeDistributed(Dense(1)))
# model.compile(optimizer='adam', loss='mse')
# fit model
# model.fit(trainX, trainY, epochs=100, verbose=0)
# demonstrate prediction
# testX = test_dataset[:3]
# testX = testX.reshape((1, 3, 1))
# y = []
# for i in range(1000):
#     # print(testX)
#     # y = model.predict(testX, verbose=0)
#     y.append(model.predict(testX, verbose=0)[0][0][0])
#     testX[0][0] = testX[0][1]
#     testX[0][1] = testX[0][2]
#     # print(y)
#     testX[0][2] = test_dataset[i+3]
# y = y.reshape(len(y), 1)
# np.savetxt("test.csv", y, delimiter=",")
# yhat = model.predict(testX, verbose=0)
# yhat = yhat.reshape(4991, 1)
# np.savetxt("test.csv", yhat, delimiter=",")
# yhat = yhat.reshape(4991, 1)
# plt.figure(figsize=(25, 3))
# plt.plot(testY)
# data = pd.read_csv('test.csv')
# plt.title('test')
# plt.plot(y)
# plt.show()
