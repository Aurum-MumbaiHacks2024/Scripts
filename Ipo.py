
# prompt: Read IPO.csv

import pandas as pd

# Assuming IPO.csv is in the current working directory
df = pd.read_csv('IPO.csv')

# Print the first 5 rows of the dataframe to verify
df.head()

import pandas as pd
from datetime import datetime

# Convert 'Date' to datetime objects
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%y')

# Calculate age as of today
today = datetime.today()
# Use .apply to perform the comparison on each row individually:
df['Age'] = df['Date'].apply(lambda x: today.year - x.year - ((today.month, today.day) < (x.month, x.day)))

# Remove rows with missing values in 'Age'
df.dropna(subset=['Age'], inplace=True)

# @title Issue_Size(crores) vs QIB

from matplotlib import pyplot as plt
df.plot(kind='scatter', x='Issue_Size(crores)', y='QIB', s=32, alpha=.8)
plt.gca().spines[['top', 'right',]].set_visible(False)

import pandas as pd
from datetime import datetime

# Convert 'Date' to datetime objects
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%y')

# Calculate age as of today
today = datetime.today()
# Using .apply to perform row-wise calculation
df['Age'] = df['Date'].apply(lambda x: today.year - x.year - ((today.month, today.day) < (x.month, x.day)))

# Remove rows with missing values in 'Age'
df.dropna(subset=['Age'], inplace=True)

!pip install statsmodels

df.head()

print(df.columns)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from tensorflow import keras
from tensorflow.keras.layers import Dense

# Assuming your DataFrame is named 'df' and contains the relevant columns

# Define target variables and predictor variables
target_vars = ['Listing_Open', 'Listing_Close']
predictor_vars = [col for col in df.columns if col not in target_vars and col != 'Date' and col != 'IPO_Name']

# Create the design matrix (X) and target matrix (Y)
X = df[predictor_vars].copy()
Y = df[target_vars]

# Convert relevant columns to numeric, handling errors
for col in X.columns:
    try:
        X.loc[:, col] = pd.to_numeric(X[col])
    except ValueError:
        print(f"Could not convert column '{col}' to numeric. It may contain non-numeric values.")
        # Handle the non-numeric column: investigate and clean 'CMP'
        if col == 'CMP':
            # Replace non-numeric with NaN
            X['CMP'] = pd.to_numeric(X['CMP'], errors='coerce')
        # You might choose to drop the column if it's not useful:
        # X = X.drop(columns=[col])

# One-hot encoding for categorical features if any
X = pd.get_dummies(X, columns=X.select_dtypes(include=['object']).columns)

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Impute missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create the neural network model
model = keras.Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Dropout(0.2),
    Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Dropout(0.2),
    Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
    Dense(2)
])

# Compile the model
optimizer = keras.optimizers.Adam(learning_rate=0.001) # Adjust learning rate
model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

# Early stopping
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10) # Stop if val_loss doesn't improve for 10 epochs

# Train the model
history = model.fit(X_train, Y_train, epochs=200, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
# Train the model

# Evaluate the model
loss, mae, mse = model.evaluate(X_test, Y_test, verbose=0)
r2 = r2_score(Y_test, model.predict(X_test))

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared: {r2}")

# You can access the training history for plotting loss and metrics:
# print(history.history.keys()) # See available keys (e.g., 'loss', 'mae', 'val_loss', 'val_mae')

# prompt: give code for exporting the above all code as pkl file

import pickle

# Assuming your code is in a variable called 'code_content'
code_content = """
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from tensorflow import keras
from tensorflow.keras.layers import Dense


# Assuming IPO.csv is in the current working directory
df = pd.read_csv('IPO.csv')

# Print the first 5 rows of the dataframe to verify
df.head()

# Convert 'Date' to datetime objects
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%y')

# Calculate age as of today
today = datetime.today()
# Use .apply to perform the comparison on each row individually:
df['Age'] = df['Date'].apply(lambda x: today.year - x.year - ((today.month, today.day) < (x.month, x.day)))

# Remove rows with missing values in 'Age'
df.dropna(subset=['Age'], inplace=True)
# @title Issue_Size(crores) vs QIB

df.plot(kind='scatter', x='Issue_Size(crores)', y='QIB', s=32, alpha=.8)
plt.gca().spines[['top', 'right',]].set_visible(False)

# Convert 'Date' to datetime objects
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%y')

# Calculate age as of today
today = datetime.today()
# Using .apply to perform row-wise calculation
df['Age'] = df['Date'].apply(lambda x: today.year - x.year - ((today.month, today.day) < (x.month, x.day)))

# Remove rows with missing values in 'Age'
df.dropna(subset=['Age'], inplace=True)
!pip install statsmodels
df.head()
print(df.columns)


# Assuming your DataFrame is named 'df' and contains the relevant columns

# Define target variables and predictor variables
target_vars = ['Listing_Open', 'Listing_Close']
predictor_vars = [col for col in df.columns if col not in target_vars and col != 'Date' and col != 'IPO_Name']

# Create the design matrix (X) and target matrix (Y)
X = df[predictor_vars].copy()
Y = df[target_vars]

# Convert relevant columns to numeric, handling errors
for col in X.columns:
    try:
        X.loc[:, col] = pd.to_numeric(X[col])
    except ValueError:
        print(f"Could not convert column '{col}' to numeric. It may contain non-numeric values.")
        # Handle the non-numeric column: investigate and clean 'CMP'
        if col == 'CMP':
            # Replace non-numeric with NaN
            X['CMP'] = pd.to_numeric(X['CMP'], errors='coerce')
        # You might choose to drop the column if it's not useful:
        # X = X.drop(columns=[col])

# One-hot encoding for categorical features if any
X = pd.get_dummies(X, columns=X.select_dtypes(include=['object']).columns)

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Impute missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create the neural network model
model = keras.Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Dropout(0.2),
    Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Dropout(0.2),
    Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
    Dense(2)
])

# Compile the model
optimizer = keras.optimizers.Adam(learning_rate=0.001) # Adjust learning rate
model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

# Early stopping
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10) # Stop if val_loss doesn't improve for 10 epochs

# Train the model
history = model.fit(X_train, Y_train, epochs=200, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
# Train the model

# Evaluate the model
loss, mae, mse = model.evaluate(X_test, Y_test, verbose=0)
r2 = r2_score(Y_test, model.predict(X_test))

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared: {r2}")

# You can access the training history for plotting loss and metrics:
# print(history.history.keys()) # See available keys (e.g., 'loss', 'mae', 'val_loss', 'val_mae')
"""

with open('IPO.pkl', 'wb') as f:
    pickle.dump(code_content, f)

print("Code exported to code.pkl")

df.columns

