import numpy as np


def gradient_descent(X, y, learning_rate=0.01, n_epochs=500):

    r, c = X.shape
    coefficients = np.zeros(c)
    intercept = 0

    for epoch in range(n_epochs):
        y_hat = intercept + np.dot(X, coefficients)
        error = y_hat - y

        p_intercept = 1 / r * np.sum(error)
        p_coefficients = 1 / r * np.sum(X.T * error)

        intercept -= learning_rate * p_intercept
        coefficients -= learning_rate * p_coefficients

        if epoch % 100 == 0:
            loss = (1 / (2 * c)) * np.sum(error**2)
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
            print(f"Intercept: {intercept:.4f}, Coefficients: {coefficients}\n")

    return intercept, coefficients


if __name__ == "__main__":

    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]], dtype=float)

    y = np.array([5, 8, 11, 14], dtype=float)

    intercept, coefficients = gradient_descent(X, y)

    print("Final intercept:", intercept)
    print("Final coefficients:", coefficients)

    preds = intercept + np.dot(X, coefficients)
    print("Predictions:", preds)
    print("True values:", y)
