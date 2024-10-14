import rnn
import perceptron
from nyse import *
import nn
import classification_performance as cp
from plotting import *
import pickle
import numpy as np

# Helper function to split data into training and testing sets
def split_data(data, split_ratio=0.5):
    n = data.x.shape[0]
    split_idx = int(n * split_ratio)
    return Data(data.x[:split_idx, :], data.y[:split_idx, :]), Data(data.x[split_idx:, :], data.y[split_idx:, :])

# Generalized function to train a neural network
def train_neural_network(model, data_train, data_test, iters, epoch=10):
    errors = {"train": [], "test": []}
    best_error = float("inf")
    
    # Train multiple models and select the best one
    for _ in range(5):
        nn_model = nn.NeuralNetwork(model, nb_epoch=20)
        error_train = nn_model.train(data_train)
        if error_train < best_error:
            best_error = error_train
            best_nn = nn_model

    nn_model = best_nn
    nn_model.nb_epoch = epoch
    
    for i in range(iters):
        print(f"--- ITERATION {i+1} / {iters} ---")
        error_train = nn_model.train(data_train)
        error_test = nn_model.test(data_test)
        
        errors["train"].append(error_train)
        errors["test"].append(error_test)

        print(f"Train ERROR: {error_train}")
        print(f"Test ERROR: {error_test}")

    return errors

# Generalized function to save results using a context manager
def save_results(filename, data):
    with open(filename, 'wb') as output:
        pickle.dump(data, output)

# Main function to train both RNN and MLP models
def main():
    input_length = 25
    hidden_cnt = 100
    cross_validation_passes = 20
    epochs = 200
    data = get_test_data(input_length)
    
    # Split data into train/test sets
    data_train, data_test = split_data(data)
    
    # Train and evaluate RNN
    print("Training RNN...")
    rnn_model = rnn.RNN(input_length, hidden_cnt, data.x.shape[2], data.y.shape[1])
    rnn_errors = train_neural_network(rnn_model, data_train, data_test, cross_validation_passes, epochs)
    
    print("Training MLP...")
    mlp_model = perceptron.MLP(input_length, hidden_cnt, data.x.shape[2], data.y.shape[1])
    mlp_errors = train_neural_network(mlp_model, data_train, data_test, cross_validation_passes, epochs)
    
    # Display performance metrics
    perf = cp.ClassificationPerformance()
    perf.add("RNN", rnn_errors["test"])
    perf.add("MLP", mlp_errors["test"])
    perf.compare()
    perf.make_plots()

    # Save results
    save_results('../../results/RNN_errors', rnn_errors)
    save_results('../../results/MLP_errors', mlp_errors)

if __name__ == '__main__':
    main()
