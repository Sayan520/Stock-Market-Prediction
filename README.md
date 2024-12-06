# Generate README.md
readme_content = """
# Tesla Stock Price Prediction

This project builds a deep learning model using **LSTM (Long Short-Term Memory)** to predict Tesla's stock prices and visualize the predictions using **Streamlit**, a Python library for creating interactive web applications.

## Features

1. **Data Preprocessing:**
   - Loads Tesla's historical stock price data from a CSV file.
   - Splits the data into training and testing sets (70% for training and 30% for testing).
   - Scales the data using **MinMaxScaler** for better performance with the LSTM model.

2. **Model Architecture:**
   - An LSTM-based neural network with:
     - Four stacked LSTM layers of varying units.
     - Dropout layers to prevent overfitting.
     - A Dense output layer for single-value predictions.
   - Optimized using the **Adam optimizer** with **mean squared error (MSE)** as the loss function.

3. **Model Evaluation:**
   - Predictions on test data are compared with actual prices using metrics like:
     - **Mean Absolute Error (MAE)**.
     - **Mean Squared Error (MSE)**.
     - **R2 Score**.
   - Visualization of results using:
     - Line chart for predicted vs. actual prices.
     - Scatter plot to visualize the correlation between predictions and actual values.
     - R2 score represented as a horizontal bar chart.

4. **Interactive Streamlit Interface:**
   - Displays Tesla's closing price trends.
   - Evaluates and visualizes model performance.
   - Allows users to input the number of future days for prediction.
   - Shows the predicted future stock prices using a line chart.

5. **Moving Average Analysis:**
   - Computes and visualizes the **100-day** and **200-day moving averages** for trend comparison.

## How to Run the Project

1. Install required libraries:
   ```bash
   pip install streamlit pandas numpy matplotlib scikit-learn tensorflow
