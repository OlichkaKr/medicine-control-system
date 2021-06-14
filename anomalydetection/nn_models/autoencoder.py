import numpy as np
from keras import callbacks
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Dropout
from keras.models import Sequential


class LSTM_NN():
    def __init__(self, optimizer='adam', loss='mae'):
        self.optimizer = optimizer
        self.loss = loss
        self.timesteps = 1
        self.n_features = 1
        self.model = None

    def create_model(self):
        model = Sequential()

        model.add(LSTM(128, input_shape=(self.timesteps, self.n_features)))
        model.add(Dropout(rate=0.2))
        model.add(RepeatVector(self.timesteps))
        model.add(LSTM(128, return_sequences=True))
        model.add(Dropout(rate=0.2))
        model.add(TimeDistributed(Dense(self.n_features)))
        model.compile(optimizer=self.optimizer, loss=self.loss)
        model.summary()

        self.model = model

    def fit(self, x, y, epochs=100, batch_size=32, validation_split=0.1):
        self.timesteps = x.shape[1]
        self.n_features = x.shape[2]
        self.create_model()

        self.model.fit(x, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split,
                       callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')], shuffle=False)

    def evaluate(self, x, y):
        self.model.evaluate(x, y)

    def predict(self, x, verbose=0):
        return self.model.predict(x, verbose=verbose)

    def find_threshold(self, x_train):
        x_train_pred = self.predict(x_train)
        train_mae_loss = np.mean(np.abs(x_train_pred - x_train), axis=1)
        return np.max(train_mae_loss)
