import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Dropout
from keras.models import Sequential
from matplotlib import pyplot as plt

train = pd.read_csv('../train_dataset.csv', names=['value', 'datetime'])
test = pd.read_csv('../test_dataset_outliers.csv', names=['value', 'datetime'])
# train, test = train_test_split(df, train_size=80 / 100)

print(train.shape, test.shape)
scaler = StandardScaler()
# scaler = scaler.fit(train[['value']])

train['value'] = scaler.fit_transform(train[['value']])
test['value'] = scaler.transform(test[['value']])
# print(train['value'].head())



TIME_STEPS = 30

def create_sequences(X, y, time_steps=TIME_STEPS):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i + time_steps)].values)
        ys.append(y.iloc[i + time_steps])

    return np.array(Xs), np.array(ys)


X_train, y_train = create_sequences(train[['value']], train['value'])
X_test, y_test = create_sequences(test[['value']], test['value'])

print(f'Training shape: {X_train.shape}')

print(f'Testing shape: {X_test.shape}')

model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(rate=0.2))
model.add(RepeatVector(X_train.shape[1]))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(rate=0.2))
model.add(TimeDistributed(Dense(X_train.shape[2])))
model.compile(optimizer='adam', loss='mae')
model.summary()

from keras import callbacks
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1,
                    callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')], shuffle=False)

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.show()

model.evaluate(X_test, y_test)

X_train_pred = model.predict(X_train, verbose=0)
train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)

plt.hist(train_mae_loss, bins=50)
plt.xlabel('Train MAE loss')
plt.ylabel('Number of Samples')

threshold = np.max(train_mae_loss)
print(f'Reconstruction error threshold: {threshold}')

X_test_pred = model.predict(X_test, verbose=0)
test_mae_loss = np.mean(np.abs(X_test_pred-X_test), axis=1)

plt.hist(test_mae_loss, bins=50)
plt.xlabel('Test MAE loss')
plt.ylabel('Number of samples')

test_score_df = pd.DataFrame(test[TIME_STEPS:])
test_score_df['loss'] = test_mae_loss
test_score_df['threshold'] = threshold
test_score_df['anomaly'] = test_score_df['loss'] > test_score_df['threshold']
test_score_df['value'] = test[TIME_STEPS:]['value']

plt.figure(figsize=(25, 3))
plt.scatter(x=test_score_df['datetime'], y=test_score_df['loss'])
plt.scatter(x=test_score_df['datetime'], y=test_score_df['threshold'])
plt.show()

anomalies = test_score_df.loc[test_score_df['anomaly'] == True]
anomalies.head()

plt.figure(figsize=(50, 3))
plt.scatter(x=test_score_df['datetime'], y=scaler.inverse_transform(test_score_df['value']))
plt.scatter(x=anomalies['datetime'], y=scaler.inverse_transform(anomalies['value']), c='r')
plt.show()

