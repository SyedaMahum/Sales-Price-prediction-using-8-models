import os
from flask import Flask, render_template, jsonify, request
import pickle
import pandas as pd
from pmdarima.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

app = Flask(__name__)

# Define paths to data files
data_directory = r'C:\Users\Thinpad\Desktop\DM_project_flask\data'
csv_file_path = os.path.join(data_directory, 'normalized_data.csv')
arima_pickle_file_path = os.path.join(data_directory, 'arima_model.pkl')
sarima_pickle_file_path = os.path.join(data_directory, 'sarima_model_1.pkl')
svr_pickle_file_path = os.path.join(data_directory, 'svr_model.pkl')
prophet_pickle_file_path =  os.path.join(data_directory, 'prophet_model.pkl')

# Check if CSV file exists
if os.path.isfile(csv_file_path):
    # Read normalized data
    normalized_df = pd.read_csv(csv_file_path)
else:
    raise FileNotFoundError(f"CSV file '{csv_file_path}' not found.")

# Check if ARIMA pickle file exists
if os.path.isfile(arima_pickle_file_path):
    # Load ARIMA model
    with open(arima_pickle_file_path, 'rb') as pkl_file:
        arima_loaded_model = pickle.load(pkl_file)
else:
    raise FileNotFoundError(f"Pickle file '{arima_pickle_file_path}' not found.")

# Check if SARIMA pickle file exists
if os.path.isfile(sarima_pickle_file_path):
    # Load SARIMA model
    with open(sarima_pickle_file_path, 'rb') as pkl_file:
        sarima_loaded_model = pickle.load(pkl_file)
else:
    raise FileNotFoundError(f"Pickle file '{sarima_pickle_file_path}' not found.")

# Check if SVR pickle file exists
if os.path.isfile(svr_pickle_file_path):
    # Load SVR model
    with open(svr_pickle_file_path, 'rb') as pkl_file:
        svr_loaded_model = pickle.load(pkl_file)
else:
    raise FileNotFoundError(f"Pickle file '{svr_pickle_file_path}' not found.")

# Check if prophet pickle file exists
if os.path.isfile(prophet_pickle_file_path):
    # Load Prophet model
    with open(prophet_pickle_file_path, 'rb') as pkl_file:
        prophet_model = pickle.load(pkl_file)
else:
    raise FileNotFoundError(f"Pickle file '{prophet_pickle_file_path}' not found.")

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Fit scaler on the data
scaler.fit(normalized_df[['Adj Close']])

# Split data into training and testing sets
train, test = train_test_split(normalized_df, train_size=0.7)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def get_data():
    # Generate predictions with ARIMA model
    predictions = arima_loaded_model.predict(n_periods=len(test))
    
    # Convert data to JSON format
    data = {
        'actual': test['Adj Close'].tolist(),
        'predicted': predictions.tolist()
    }
    
    return jsonify(data)

   
@app.route('/sarima_data')
def get_sarima_data():
    # Make predictions on the test set using the SARIMA model
    sarima_predictions = sarima_loaded_model.predict(start=len(train), end=len(train) + len(test) - 1)

    # Convert data to JSON format
    data = {
        'actual': test['Adj Close'].tolist(),
        'predicted': sarima_predictions.tolist()
    }

    return jsonify(data)

@app.route('/svr_data')
def get_svr_data():
    # Predict with SVR model
    X_test = test.index.values.reshape(-1, 1)
    y_pred = svr_loaded_model.predict(X_test)

    # Convert data to JSON format
    data = {
        'actual': test['Adj Close'].tolist(),
        'predicted': y_pred.tolist()
    }

    return jsonify(data)
# Route to make predictions using the loaded Prophet model
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the selected model from the request
        selected_model = request.form.get('model')

        # Check if the selected model is available and if it's the Prophet model
        if selected_model == 'Prophet':
            try:
                # Make predictions using the loaded Prophet model
                forecast = prophet_model.predict()
                # Extract relevant columns from the forecast
                predictions = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                # Render the visualization template with predictions as context data
                return render_template('visualization.html', predictions=predictions)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        else:
            return jsonify({'error': 'Selected model is not available'}), 400
    elif request.method == 'GET':
        try:
            # Make predictions using the loaded Prophet model
            forecast = prophet_model.predict()
            # Extract relevant columns from the forecast
            predictions = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            # Convert predictions to JSON and return
            return predictions.to_json()
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        



ets_pickle_file_path = os.path.join(data_directory, 'ets_model.pkl')
if os.path.isfile(ets_pickle_file_path):
    # Load ETS model
    with open(ets_pickle_file_path, 'rb') as pkl_file:
        ets_model = pickle.load(pkl_file)
else:
    raise FileNotFoundError(f"Pickle file '{ets_pickle_file_path}' not found.")
def fit_ets_model(data):
    model = ExponentialSmoothing(data, trend='add', seasonal='add', seasonal_periods=12)
    fit_model = model.fit()
    predictions = fit_model.forecast(steps=len(data))
    return predictions

# Generate predictions with the loaded ETS model
predictions = fit_ets_model(normalized_df['Adj Close'])

# Define Flask route to serve ETS data
@app.route('/ets_data')
def get_ets_data():
    data_dict = {
        'actual': normalized_df['Adj Close'].tolist(),
        'predicted': predictions.tolist()
    }
    return jsonify(data_dict)


if __name__ == '__main__':

    app.run(debug=True)
