import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from collision_utils import *

# Load processed data
X_scaled = np.load('X_scaled.npy')
y = np.load('y.npy')

print(f"X_scaled shape: {X_scaled.shape}")
print(f"y shape: {y.shape}")

# Train the model
print("Training model...")
w, b, costs = gradient_descent(X_scaled, y, alpha=0.1, iterations=1000)

# Save the model
np.save('weights.npy', w)
np.save('bias.npy', np.array([b]))
print("Model saved!")

# Make predictions
y_pred_prob = sigmoid(X_scaled @ w + b)
y_pred = (y_pred_prob >= 0.5).astype(int)

# Accuracy
accuracy = np.mean(y_pred == y)
print(f"Training Accuracy: {accuracy:.4f}")

# Confusion Matrix
print(confusion_matrix(y, y_pred))
print(classification_report(y, y_pred))