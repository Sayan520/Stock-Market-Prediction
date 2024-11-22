import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf

# Load the data
data = pd.read_csv('C:/Users/ghosh/Downloads/TSLA Training.csv')

# Prepare data 
train = pd.DataFrame(data[0:int(len(data) * 0.70)])
test = pd.DataFrame(data[int(len(data) * 0.70): int(len(data))])
scaler = MinMaxScaler(feature_range=(0, 1))
train_close = train.iloc[:, 4:5].values
test_close = test.iloc[:, 4:5].values
data_training_array = scaler.fit_transform(train_close)

# Prepare training data
x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100: i])
    y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build the model
model = Sequential()
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=30, activation='relu', return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(units=120, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.metrics.MeanAbsoluteError()])
model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1)

# Prepare test data
past_100_days = pd.DataFrame(train_close[-100:])
test_df = pd.DataFrame(test_close)
final_data = pd.concat([past_100_days, test_df], ignore_index=True)
input_data = scaler.transform(final_data)
x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)

# Predictions
y_pred = model.predict(x_test)
scale_factor = 1/0.00331268
y_pred = y_pred * scale_factor
y_test = y_test * scale_factor

# Streamlit app
st.title('Tesla Stock Price Prediction')
st.write('## Closing Price Of Tesla')
st.line_chart(data['Close'])

st.write('## Model Evaluation')
mae = mean_absolute_error(y_test, y_pred)
mae_percentage = (mae / np.mean(y_test)) * 100
st.write(f"Mean Absolute Error: {mae_percentage:.2f}%")

mse = mean_squared_error(y_test, y_pred)
st.write(f"Mean Squared Error: {mse}")

r2 = r2_score(y_test, y_pred)
st.write(f"R2 Score: {r2:.2f}")

fig, ax = plt.subplots()
ax.barh(0, r2, color='skyblue')
ax.set_xlim([-1, 1])
ax.set_yticks([])
ax.set_xlabel('R2 Score')
ax.set_title('R2 Score')
ax.text(r2, 0, f'{r2:.2f}', va='center', color='black')
st.pyplot(fig)

fig, ax = plt.subplots()
ax.plot(y_test, 'g', label="Original Price")
ax.plot(y_pred, 'r', label="Predicted Price")
ax.set_title('Tesla Stock Price Prediction')
ax.set_xlabel('Time')
ax.set_ylabel('Price')
ax.legend()
st.pyplot(fig)

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.plot([min(y_test), max(y_test)], [min(y_pred), max(y_pred)], 'r--')
ax.set_xlabel('Actual Values')
ax.set_ylabel('Predicted Values')
ax.set_title(f'R2 Score: {r2:.2f}')
st.pyplot(fig)

# Future prediction
st.write('## Future Prediction')
days_to_predict = st.number_input('Enter number of days to predict', min_value=1, max_value=365, value=30)
future_data = input_data[-365:]
future_predictions = []
for _ in range(days_to_predict):
    future_input = np.reshape(future_data, (1, future_data.shape[0], 1))
    future_pred = model.predict(future_input)
    future_predictions.append(future_pred[0][0])
    future_data = np.append(future_data[1:], future_pred[0])

future_predictions = np.array(future_predictions) * scale_factor
st.write('Predicted Future Prices:')
st.line_chart(future_predictions)

# Calculate 100-day and 200-day moving averages
data['MA100'] = data['Close'].rolling(100).mean()
data['MA200'] = data['Close'].rolling(200).mean()

# Plotting moving averages in Streamlit
st.write("## Comparison of 100-day and 200-day Moving Averages")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data['Close'], label='Close Price')
ax.plot(data['MA100'], 'r', label='100-day MA')
ax.plot(data['MA200'], 'g', label='200-day MA')
ax.grid(True)
ax.set_title('Comparison of 100-day and 200-day Moving Averages')
ax.legend()
st.pyplot(fig)
