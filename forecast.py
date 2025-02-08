#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 11:38:14 2025

@author: soundaryarupavataram
"""

## Import Libraries
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import keras_tuner as kt
from prophet import Prophet
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
#keras = tf.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Flatten, Dropout, MaxPooling1D, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler

np.random.seed(42)
tf.random.set_seed(42)

# Function to create sequences for LSTM
def create_sequences(data, lookback=10):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback])
    return np.array(X), np.array(y)

def build_n_save_model(X_train, y_train, X_val, y_val):
    
    model_name = 'model_cnn_lstm_v1' + '-' + str(datetime.now().strftime("%Y-%m-%d")) + '.h5'
    
    # Create CNN-LSTM Model
    model_cnn_lstm = Sequential([
        Input(shape=(lookback, 1)),
        Conv1D(filters = 4, kernel_size = 2, activation = 'relu'),
        LSTM(100, activation='relu', return_sequences=True),
        LSTM(100, activation='relu'),
        Dense(10),
        #Dropout(0.2),
        Dense(1)
    ])

    # Compile Model
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
    #cnn_lstm_checkpoint = keras.callbacks.ModelCheckpoint(model_name, save_best_only = True, save_weights_only=True)
    model_cnn_lstm.compile(optimizer = optimizer, loss = tf.keras.losses.MeanSquaredError(), metrics = ['mae'])
    
    # Train the model
    model_cnn_lstm.fit(
        X_train, y_train,
        epochs = 100, batch_size = 16,
        validation_data = (X_val, y_val),
        verbose = 0,
        #callbacks = [cnn_lstm_checkpoint]
    )
    
    return model_cnn_lstm


if __name__ == "__main__":
    
    start = datetime.now()
    #############################  Load data  ##############################################
    
    cases_df = pd.read_csv('data/covid19_daily_data.csv')
    
    # Convert date column to datetime
    cases_df["Date_reported"] = pd.to_datetime(cases_df["Date_reported"])
    
    # Aggregate New Cases for Global Infection Rate
    daily_cases = cases_df.groupby("Date_reported")["New_cases"].sum().reset_index()
    
    # Replace all -1 values with 0
    daily_cases['New_cases'] = daily_cases['New_cases'].replace(-1, 0)
    
    print('Sample size:', len(daily_cases))
    
    #############################  Data Preprocessing ##############################################
    preprocess_start = datetime.now()
    lookback = 10
    
    # Split data into Training and Test Sets
    train_size = int(len(daily_cases) * 0.8)
    
    train_data = daily_cases['New_cases'][:train_size].values.reshape(-1, 1)
    test_data = daily_cases['New_cases'][train_size:].values.reshape(-1, 1)
    
    # Apply the MinMaxScaler on train and test
    scaler = MinMaxScaler(feature_range=(0, 1))  
    scaler.fit(train_data)  # Fit only on training data
    
    scaled_train_data = scaler.transform(train_data)
    
    # Prepare Train & Validation Data
    X, y = create_sequences(scaled_train_data, lookback)
    val_size = int(len(X) * 0.8)
    
    X_train, y_train = X[:val_size], y[:val_size]
    X_val, y_val = X[val_size:], y[val_size:]
    
    preprocess_end = datetime.now()
    print('Time to pre process:', preprocess_end - preprocess_start)

    ##############################  Model Training  ############################
    
    train_start = datetime.now()
    model_cnn_lstm = build_n_save_model(X_train, y_train, X_val, y_val)
    train_end = datetime.now()
    print('Time to train model:', train_end - train_start)
    
    ##############################  Model Prediction  #########################
    
    pred_start_time = datetime.now()
    
    scaled_test_data = scaler.transform(test_data)
    
    # Prepare Test Data
    X_test, y_test = create_sequences(scaled_test_data, lookback)
    
    # Inverse Transform
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Predict on Validation and Test Sets
    y_pred = model_cnn_lstm.predict(X_test)
    
    pred_end_time = datetime.now()
    print('total prediction script time: ', pred_end_time - pred_start_time)
    
    ###########################  Metrics  #####################################
    
    import numpy as np
    
    # Assuming y_test and y_pred are NumPy arrays (inverse-transformed to original scale)
    y_test = y_test.flatten()
    y_pred = y_pred.flatten()
    
    # Compute Metrics
    metrics_dict = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2_Score": r2_score(y_test, y_pred),
        "MAPE": np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    }
    
    # Print Metrics Dictionary
    print("CNN-LSTM Model Evaluation Metrics:")
    for key, value in metrics_dict.items():
        print(f"{key}: {value:.4f}")
    
    end = datetime.now()
    
    print('total script time: ', end - start)
    
    
