#!/usr/bin/env python
# coding: utf-8

#  ## LSTM Stock Predictor Using Closing Prices

# #### In this notebook, you will build and train a custom LSTM RNN that uses a 10 day window of Bitcoin closing prices to predict the 11th day closing price.
# 
# You will need to:
# 
#     Prepare the data for training and testing
#     Build and train a custom LSTM RNN
#     Evaluate the performance of the model
# 

# ### Data Preparation
# 
# #### In this section, you will need to prepare the training and testing data for the model. The model will use a rolling 10 day window to predict the 11th day closing price.
# 
# You will need to:
# 
#     Use the window_data function to generate the X and y values for the model.
#     Split the data into 70% training and 30% testing
#     Apply the MinMaxScaler to the X and y values
#     Reshape the X_train and X_test data for the model. Note: The required input format for the LSTM is:
# 
# reshape((X_train.shape[0], X_train.shape[1], 1))
# 

# In[62]:


pwd downloads


# In[63]:


import numpy as np
import pandas as pd
import hvplot.pandas


# In[64]:


# Set the random seed for reproducibility
# Note: This is for the homework solution, but it is good practice to comment this out and run multiple experiments to evaluate your model
from numpy.random import seed
seed(1)
from tensorflow import random
random.set_seed(2)


# In[65]:


# Load the fear and greed sentiment data for Bitcoin
df = pd.read_csv('btc_sentiment.csv', index_col="date", infer_datetime_format=True, parse_dates=True)
df = df.drop(columns="fng_classification")
df.head()


# In[66]:


# Load the historical closing prices for bitcoin
df2 = pd.read_csv('btc_historic.csv', index_col="Date", infer_datetime_format=True, parse_dates=True)['Close']
df2 = df2.sort_index()
df2.head()


# In[67]:


df2.tail()


# In[68]:


# Join the data into a single DataFrame
df3 = df.join(df2, how="inner")
df3.tail()


# In[69]:


df3.head()


# In[111]:


# This function accepts the column number for the features (X) and the target (y)
# It chunks the data up with a rolling window of Xt-n to predict Xt
# It returns a numpy array of X any y
def window_data(df3, window, feature_col_number, target_col_number):
    X = []
    y = []
    for i in range(len(df3) - window - 1):
        features = df3.iloc[i:(i + window), feature_col_number]
        target = df3.iloc[(i + window), target_col_number]
        X.append(features)
        y.append(target)
    return np.array(X), np.array(y).reshape(-1, 1)


# In[112]:


# Predict Closing Prices using a 10 day window of previous closing prices
# Try a window size anywhere from 1 to 10 and see how the model performance changes
window_size = 1

# Column index 1 is the `Close` column
feature_column = 1
target_column = 1
X, y = window_data(df3, window_size, feature_column, target_column)


# In[113]:


# Use 70% of the data for training and the remaineder for testing
# YOUR CODE HERE!
split = int(0.7 * len(x))
X_train = X[: split - 1]
X_test = X[split:]
y_train = y[: split - 1]
y_test = y[split:]


# In[114]:


# Use MinMaxScaler to scale the data between 0 and 1. 
# YOUR CODE HERE!
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
scaler.fit(y)
y_train = scaler.transform(y_train)
y_test = scaler.transform(y_test)


# In[115]:


# Reshape the features for the model
# YOUR CODE HERE!
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))


#  ### Build and Train the LSTM RNN
# #### In this section, you will design a custom LSTM RNN and fit (train) it using the training data.You will need to:
# #### Define the model architectureCompile the modelFit the model to the training data
# Hints:
# You will want to use the same model architecture and random seed for both notebooks. This is necessary to accurately compare the performance of the FNG model vs the closing price model. 

# In[116]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


# In[117]:


# Build the LSTM model. 
# The return sequences need to be set to True if you are adding additional LSTM layers, but 
# You don't have to do this for the final layer. 
# YOUR CODE HERE!
model = Sequential()

number_units = 5
dropout_fraction = 0.2

# Layer 1
model.add(LSTM(
    units=number_units,
    return_sequences=True,
    input_shape=(X_train.shape[1], 1))
    )
model.add(Dropout(dropout_fraction))
# Layer 2
model.add(LSTM(units=number_units, return_sequences=True))
model.add(Dropout(dropout_fraction))
# Layer 3
model.add(LSTM(units=number_units))
model.add(Dropout(dropout_fraction))
# Output layer
model.add(Dense(1))


# In[118]:


# Compile the model
# YOUR CODE HERE!
model.compile(optimizer="adam", loss="mean_squared_error")


# In[119]:


# Summarize the model
model.summary()


# In[120]:


# Train the model
# Use at least 10 epochs
# Do not shuffle the data
# Experiement with the batch size, but a smaller batch size is recommended
model.fit(X_train, y_train, epochs=10, shuffle=False, batch_size=10, verbose=1)


# #### Model Performance
# 
# In this section, you will evaluate the model using the test data.
# 
# You will need to:
# 
#     Evaluate the model using the X_test and y_test data.
#     Use the X_test data to make predictions
#     Create a DataFrame of Real (y_test) vs predicted values.
#     Plot the Real vs predicted values as a line chart
# 
# Hints
# 
# Remember to apply the inverse_transform function to the predicted and y_test values to recover the actual closing prices.
# 

# In[121]:


# Evaluate the model
model.evaluate(X_test, y_test)


# In[122]:


# Make some predictions
# YOUR CODE HERE!
predicted = model.predict(X_test)


# In[123]:


# Recover the original prices instead of the scaled version
predicted_prices = scaler.inverse_transform(predicted)
real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))


# In[124]:


Stocks = pd.DataFrame({
    "Real": real_prices.ravel(),
    "Predicted": predicted_prices.ravel()
}, index = df3.index[-len(real_prices): ]) 
stocks.head()


# In[126]:


# Plot the real vs predicted values as a line chart
Stocks.hvplot(title="Real vs. Predicted Stock Price", xlabel="Days", ylabel="Price")


# ### Unless I messed something up up top the Real is a lot higher and also more volatile than the predicted. 
# #### I always look up/double check how to make sure my graphs are labeled well. Labeled graphs are important

# In[ ]:




