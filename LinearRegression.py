import numpy as np
import random


def SimpleTrick(intercept, coefficients, x, y):

    y_hat = intercept + np.dot(coefficients, x)

    for i in range(len(coefficients)):
        small_rand = random.random() * 0.1

        if y_hat < y and x[i] > 0:
            coefficients[i] += small_rand
            intercept += small_rand
        elif y_hat < y and x[i] < 0:
            coefficients[i] -= small_rand
            intercept += small_rand
        elif y_hat > y and x[i] < 0:
            coefficients[i] -= small_rand
            intercept -= small_rand
        elif y_hat > y and x[i] > 0:
            coefficients[i] += small_rand
            intercept -= small_rand

    return intercept, coefficients


def AbsTrick(intercept, coefficients, x, y, learning_rate=0.01):

    y_hat = intercept + np.dot(coefficients, x)
    error = y - y_hat

    if error > 0:
        intercept += learning_rate * error
        coefficients += learning_rate * x
    elif error < 0:
        intercept -= learning_rate * error
        coefficients -= learning_rate * x

    return intercept, coefficients


def SquareTrick(intercept, coefficients, x, y, learning_rate=0.01):

    y_hat = intercept + np.dot(coefficients, x)
    error = y - y_hat

    intercept = intercept + learning_rate * error
    coefficients = coefficients + learning_rate * x * error

    return intercept, coefficients


if __name__ == "__main__":

    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]], dtype=float)

    y = np.array([5, 8, 11, 14], dtype=float)

    n_features = X.shape[1]
    coefficients = np.zeros(n_features)
    intercept = 0.0

    for epoch in range(500):
        for i in range(len(X)):
            intercept, coefficients = SquareTrick(intercept, coefficients, X[i], y[i])

    print("Trained intercept:", intercept)
    print("Trained coefficients:", coefficients)

    preds = intercept + np.dot(X, coefficients)
    print("Predictions:", preds)
    print("True:", y)
