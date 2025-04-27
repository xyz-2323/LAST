import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

data=pd.read_csv(r"C:\Users\Admin\Downloads\housing_data (1) - housing_data (1).csv")
# 3. Data Exploration and Preprocessing
data.isnull().sum()
data.fillna(data.mean(), inplace=True)

print(data.describe())
print(data.info())

# 4. Data Visualization
sns.distplot(data['MEDV'])
plt.title('Distribution of House Prices')
plt.show()

sns.boxplot(x=data['MEDV'])
plt.title('Boxplot of House Prices')
plt.show()

plt.figure(figsize=(15,12))
sns.heatmap(data.corr(), annot=True, square=True)
plt.title('Feature Correlation Heatmap')
plt.show()

important_features = ['LSTAT', 'RM', 'PTRATIO']
plt.figure(figsize=(20,5))
for idx, feature in enumerate(important_features):
    plt.subplot(1, 3, idx+1)
    plt.scatter(data[feature], data['MEDV'])
    plt.title(f'{feature} vs MEDV')
    plt.xlabel(feature)
    plt.ylabel('House Price (in $1000s)')
plt.show()

# 5. Prepare Data for Modeling
X = data.drop('MEDV', axis=1)
y = data['MEDV']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

y_pred_lr = lr_model.predict(X_test_scaled)

mse_lr = mean_squared_error(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print(f"Linear Regression - MSE: {mse_lr:.2f}, MAE: {mae_lr:.2f}, R2 Score: {r2_lr:.2f}")

# 7. Neural Network Model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model.summary()

# 8. Train Neural Network
history = model.fit(X_train_scaled, y_train, epochs=100, validation_split=0.1, verbose=1)

# 9. Plot Training History
plt.figure(figsize=(10,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss Over Epochs')
plt.show()

plt.figure(figsize=(10,5))
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.title('Training and Validation MAE Over Epochs')
plt.show()

# 10. Evaluate Neural Network
mse_nn, mae_nn = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Neural Network - MSE: {mse_nn:.2f}, MAE: {mae_nn:.2f}")

# 11. Predict New Data
new_data = np.array([[0.1, 10.0, 5.0, 0, 0.4, 6.0, 50, 6.0, 1, 400, 20, 300, 10]])
new_data_scaled = scaler.transform(new_data)

predicted_price = model.predict(new_data_scaled)
print(f"Predicted House Price: {predicted_price[0][0]:.2f} (in $1000s)")
