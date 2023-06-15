import numpy as np

# Function to initialize W1, W2, b1, b2
def initialize_parameters(n_classes, n_features):
    W1 = np.random.randn(n_classes, n_features) * 0.01
    b1 = np.zeros((n_classes, 1))
    W2 = np.random.randn(n_classes, n_classes) * 0.01
    b2 = np.zeros((n_classes, 1))
    return W1, b1, W2, b2


# Forward propagation function
def forward(X, W1, b1, W2, b2):
    Z1 = np.dot(W1, X) + b1
    A1 = np.maximum(0, Z1)  # ReLU activation
    Z2 = np.dot(W2, A1) + b2
    A2 = np.exp(Z2) / np.sum(np.exp(Z2), axis=0)  # Softmax activation
    return A1, A2


# One-hot encoding function
def one_hot(y, n_classes=10):
    one_hot = np.zeros((n_classes, y.shape[0]))
    for i in range(y.shape[0]):
        one_hot[y[i], i] = 1
    return one_hot


# Backward propagation function
def backward(X, Y, A1, A2, W1, W2):
    m = X.shape[1]
    dZ2 = A2 - one_hot(Y)
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))  # ReLU derivative
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    return dW1, db1, dW2, db2

# Update parameters function
def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return W1, b1, W2, b2


# Get prediction function
def get_prediction(X, W1, b1, W2, b2):
    A1, A2 = forward(X, W1, b1, W2, b2)
    return np.argmax(A2, axis=0)


# Get accuracy function
def get_accuracy(X, Y, W1, b1, W2, b2):
    pred = get_prediction(X, W1, b1, W2, b2)
    accuracy = np.mean(pred == Y)
    return accuracy


# Train function (gradient descent)
def train(X, Y, n_iterations=500, learning_rate=0.1):
    W1, b1, W2, b2 = initialize_parameters(n_classes, n_features)
    for i in range(n_iterations):
        A1, A2 = forward(X, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = backward(X, Y, A1, A2, W1, W2)
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
        if i % 10 == 0:
            accuracy = get_accuracy(X, Y, W1, b1, W2, b2)
            print("Iteration:", i)
            print("Accuracy:", accuracy)
    return W1, b1, W2, b2