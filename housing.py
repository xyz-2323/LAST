import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset
df = pd.read_csv('boston_housing.csv')  # Replace with your actual file path

dataset.isnull().sum()
dataset.fillna(dataset.mean(), inplace=True)
# Separate features (X) and target (Y)
X = df.drop('MEDV', axis=1)
Y = df['MEDV']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

# Normalize the data
X_train = preprocessing.normalize(X_train)
X_test = preprocessing.normalize(X_test)

# Define the neural network model
def house_prediction_model():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # No activation for output layer
    model.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])
    return model

# Instantiate and train the model
model = house_prediction_model()
history = model.fit(X_train, Y_train, epochs=100, batch_size=32, verbose=1, validation_data=(X_test, Y_test))

# Predict and evaluate the model
predictions = model.predict(X_test)
mse_nn, mae_nn = model.evaluate(X_test, Y_test)

# Print results
print("Mean Squared Error: ", mse_nn)
print("Mean Absolute Error: ", mae_nn)
