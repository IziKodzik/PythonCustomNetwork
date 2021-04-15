import numpy as np


class NetworkLayer:
    # for now its for sigmo func but I will add possibility to change activation functions in future

    def activation_function(self, x):
        return 1 / (1 + np.exp(-x))

    def activation_function_derivative(self, x):
        return self.activation_function(x) * (1 - self.activation_function(x))

    def calculate_errors(self, activation_values, output_values, expected_values):
        return self.activation_function_derivative(activation_values) * (expected_values - output_values)

    def __init__(self, weights, thresholds):
        self.weights = weights
        self.thresholds = thresholds
        self.learning_const = 0.4

    def get_activation_values(self, inputs):
        return np.matmul(self.weights, inputs)

    def get_responses(self, inputs):
        return self.activation_function(self.get_activation_values(inputs))

    def get_responses_from_activation_values(self, activation_values):
        return self.activation_function(activation_values)

    def present(self):
        print(40 * '===============')
        print('Weights:')
        print(self.weights)
        print('Thresholds:')
        print(self.thresholds)
        print(40 * '===============')

    @staticmethod
    def generate_layer(input_size, output_size):
        weights = np.random.rand(output_size, input_size)
        thresholds = np.random.rand(output_size) * (-1)
        return NetworkLayer(weights, thresholds)


class NeuralNetwork:

    def __init__(self, layers):
        self.layers = layers

    def get_responses(self, inputs):
        modified_inputs = inputs
        for layer in self.layers:
            modified_inputs = layer.get_responses(modified_inputs)
        return modified_inputs

    def teach_layers(self, first_input, expected_outputs):
        inputs = [first_input]
        a_values = []
        for index in range(len(self.layers)):
            a_values.append(self.layers[index].get_activation_values(inputs[index]))
            inputs.append(self.layers[index].get_responses_from_activation_values(a_values[index]))

        layer = self.layers[-1]
        all_errors = []
        errors = layer.activation_function_derivative(a_values[-1]) * (expected_outputs - inputs[-1])
        all_errors.append(errors)

        for i in range(len(self.layers)-2, -1, -1):
            layer = self.layers[i]
            next_layer = self.layers[i+1]
            r_i = len(self.layers) - 2 - i
            errors = layer.activation_function_derivative(a_values[i]) * np.matmul(all_errors[r_i], next_layer.weights)
            all_errors.append(errors)

        all_errors.reverse()
        # print(all_errors[-1])

        for i in range(len(self.layers)):
            errors = all_errors[i]

            weights_delta = np.zeros((len(errors), len(inputs[i])))
            for j in range(len(errors)):
                weights_delta[j] = errors[j] * inputs[i]
            self.layers[i].weights += self.layers[i].learning_const * weights_delta

        return inputs[-1]

    @staticmethod
    def generate_network(input_sizes):
        layers = []
        for index in range(len(input_sizes) - 1):
            layers.append(NetworkLayer.generate_layer(input_sizes[index], input_sizes[index + 1]))
        return NeuralNetwork(layers)

    def present(self):
        for layer in self.layers:
            layer.present()

