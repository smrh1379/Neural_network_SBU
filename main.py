import copy

import matplotlib.pyplot as plt
import numpy as np
from numpy import tanh
from sklearn import datasets
from sklearn.model_selection import train_test_split


x, y = datasets.make_moons(n_samples=1000, shuffle=True, noise=0.1, random_state=None)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
hidden_layer_activations = {0: "sigmoid", 1: "tanh"}
output_layer_activations = {0: "sigmoid", 1: "tanh", 2: "linear"}
hidden_activation = hidden_layer_activations.get(0)  # choosing activation function of hidden layer
output_activation = output_layer_activations.get(0)  # choosing activation function of hidden layer
learning_processes = {0: "Batch", 1: "Online"}
learning = learning_processes.get(0)  # choosing learning process
Use_momentum = False
Use_decay_rate = False


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tan_h(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def initialize(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x) * 0.001
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.001
    b2 = np.zeros((n_y, 1))
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1) if hidden_activation == "sigmoid" else tan_h(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2) if output_activation == "sigmoid" else tan_h(Z2) if output_activation == "tanh" else Z2

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache


def compute_cost(y_hat, y_star):
    m = y_star.shape[1] if learning == "Batch" else 1
    cost = (y_star - y_hat) ** 2
    cost = cost.sum() / m
    cost = float(np.squeeze(cost))

    return cost


def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]

    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]

    error = (A2 - Y) / m
    if output_activation == "sigmoid":
        derivation2 = A2*(1 - A2)
    if output_activation == "tanh":
        derivation2 = 1 - (A2**2)
    if output_activation == "linear":
        derivation2 = np.ones(error.shape)
    dZ2 = error * derivation2
    dW2 = np.dot(dZ2, A1.T)
    db2 = np.sum(dZ2, axis=1, keepdims=True)
    if hidden_activation == "sigmoid":
        derivation1 = (A1*(1 - A1))
    if hidden_activation == "tanh":
        derivation1 = (1 - (A1**2))
    dZ1 = (np.dot(W2.T, dZ2)* derivation1)
    dW1 = np.dot(dZ1, X.T)
    db1 = np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads


def update_parameters(parameters, grads, learning_rate, momentum):
    beta = 0.5
    W1 = copy.deepcopy(parameters["W1"])
    W2 = copy.deepcopy(parameters["W2"])
    b1 = copy.deepcopy(parameters["b1"])
    b2 = copy.deepcopy(parameters["b2"])

    dW1 = copy.deepcopy(grads["dW1"])
    db1 = copy.deepcopy(grads["db1"])
    dW2 = copy.deepcopy(grads["dW2"])
    db2 = copy.deepcopy(grads["db2"])
    v_dw1 = beta * momentum[0] + (1 - beta) * dW1
    v_dw2 = beta * momentum[1] + (1 - beta) * dW2
    v_db1 = beta * momentum[2] + (1 - beta) * db1
    v_db2 = beta * momentum[3] + (1 - beta) * db2
    if Use_momentum:
        # With momentum
        W1 = W1 - learning_rate * v_dw1
        W2 = W2 - learning_rate * v_dw2
        b1 = b1 - learning_rate * v_db1
        b2 = b2 - learning_rate * v_db2
    else:
        # Without momentum
        W1 = W1 - learning_rate * dW1
        W2 = W2 - learning_rate * dW2
        b1 = b1 - learning_rate * db1
        b2 = b2 - learning_rate * db2

    momentum = [v_dw1, v_dw2, v_db1, v_db2]
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters, momentum


def nn_model(X, Y, n_h, num_iterations, learning_rate, lr_decay=1.0, print_cost=True, learning_bias=1):
    np.random.seed(3)
    n_x = X.shape[0]
    n_y = Y.shape[0]
    initial_learning_rate = learning_rate
    momentum = [0, 0, 0, 0]
    costs = []
    if learning == "Batch":
        parameters = initialize(n_x, n_h, n_y)
        for i in range(0, num_iterations):
            A2, cache = forward_propagation(X, parameters)
            cost = compute_cost(A2, Y)
            costs.append(cost)
            grads = backward_propagation(parameters, cache, X, Y)
            parameters, momentum = update_parameters(parameters, grads, learning_rate, momentum)
            if i != 0 and i % 1000 == 0 and Use_decay_rate:
                learning_rate = learning_rate * lr_decay
                # learning_rate =initial_learning_rate*(1/(1+lr_decay*i))
                # learning_rate = (lr_decay ** i) * initial_learning_rate
            # if cost < learning_bias:
            #     break
            if print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost))
                # if i > 1000:
                #     diff = 0
                #     for j in (reversed(range(len(costs) - 15, len(costs)))):
                #         diff += abs(costs[j] - costs[j - 1])
                #     if diff < 0.0001 and costs[len(costs) - 1] < 0.01:
                #         break
    if learning == "Online":
        parameters = initialize(n_x, n_h, n_y)
        X = np.array(X)
        Y = np.array(Y)
        for epoch in range(0, num_iterations):
            cost = 0
            temp = ([[X[:, i], Y[:, i]] for i in range(X.shape[1])])
            np.random.shuffle(temp)
            for i in range(X.shape[1]):
                temp_x = temp[i][0]
                temp_x = np.reshape(temp_x, (2, 1))
                temp_y = np.array(temp[i][1][0])
                A2, cache = forward_propagation(temp_x, parameters)
                cost += compute_cost(A2, temp_y)
                grads = backward_propagation(parameters, cache, temp_x, temp_y)
                parameters, momentum = update_parameters(parameters, grads, learning_rate, momentum)
            costs.append(cost/i)
            if cost/i < learning_bias:
                print("Cost after iteration %i: %f" % (epoch, cost/i))
                break
            if print_cost and epoch % 1000 == 0:
                print("Cost after iteration %i: %f" % (epoch, cost/i))
            if epoch != 0 and epoch % 100 == 0 and Use_decay_rate:
                learning_rate = learning_rate * lr_decay

            # if epoch > 1000:
            #     diff = 0
            #     for j in (reversed(range(len(costs) - 15, len(costs)))):
            #         diff += abs(costs[j] - costs[j - 1])
            #     if diff < 0.0001 and costs[len(costs) - 1] < 0.01:
            #         break
    plot_costs(costs, learning_rate)

    return parameters


def plot_costs(costs, learning_rate):
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()


def predict(parameters, X):
    A2, cache = forward_propagation(X, parameters)
    predictions = (A2)>0.5
    return predictions

def plot(model, X, y):
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)


if __name__ == '__main__':
    y_train = y_train.reshape(1, y_train.shape[0])
    param = nn_model(X_train.T, y_train, n_h=7, num_iterations=100000, learning_rate=0.75, lr_decay=0.99,
                     learning_bias=0.0007)
    X, Y = X_test.T, y_test.reshape(1, y_test.shape[0])
    plot(lambda x: predict(param, x.T), X, Y)
    plt.show()