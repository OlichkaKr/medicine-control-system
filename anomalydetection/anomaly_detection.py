import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from nn_models.autoencoder import LSTM_NN
from utils.data_preprocessing import DataPreprocesser

TIME_STEPS = 30

train = pd.read_csv('utils/data/train_dataset.csv', names=['value', 'datetime'])
test = pd.read_csv('utils/data/test_dataset_outliers.csv', names=['value', 'datetime'])

preprocesser = DataPreprocesser()

train['value'] = preprocesser.scale_standard(train[['value']])
test['value'] = preprocesser.scale_standard(test[['value']])

X_train, y_train = preprocesser.create_sequences(train[['value']], train['value'])
X_test, y_test = preprocesser.create_sequences(test[['value']], test['value'])

lstm = LSTM_NN()
lstm.fit(X_train, y_train)
lstm.evaluate(X_train, y_train)
threshold = lstm.find_threshold(X_train)
print(f'Reconstruction error threshold: {threshold}')

X_test_pred = lstm.predict(X_test, verbose=0)
test_mae_loss = np.mean(np.abs(X_test_pred-X_test), axis=1)

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
print(anomalies.head())

plt.figure(figsize=(25, 3))
plt.scatter(x=test_score_df['datetime'], y=preprocesser.inverse_scale_standard(test_score_df['value']))
plt.scatter(x=anomalies['datetime'], y=preprocesser.inverse_scale_standard(anomalies['value']), c='r')
plt.show()
