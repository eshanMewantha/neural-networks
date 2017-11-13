import numpy as np


class Perceptron:
    def __init__(self, number_of_inputs, learning_rate):
        self.weights = np.random.rand(1, number_of_inputs + 1)[0]
        self.learning_rate = learning_rate

    """A step function where non-negative values are returned by a 1 and negative values are returned by a -1"""
    def activate(self, z):
        if z >= 0:
            return 1
        else:
            return -1

    def feed_forward(self, input_values):
        inputs = np.array([
            input_values[0], input_values[1], -1
        ])
        z = inputs.dot(self.weights.transpose())
        return self.activate(z)

    def update_weights(self, actual_x, error):
        x = np.array([
            actual_x[0], actual_x[1], -1
        ])
        self.weights += self.learning_rate*error*x

"""
Below code simulates a perceptron learning to act as an AND gate.
(-1) represents 0
(+1) represents 1

"""
if __name__ == "__main__":
    print("\nPerceptron learning the AND gate functionality\n")
    np.random.seed(1111)
    perceptron = Perceptron(2, 0.01)
    training_x = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    training_y = np.array([[-1], [-1], [-1], [1]])

    for epoch in range(25):
        total_error = 0
        for example in range(len(training_x)):
            y_predicted = perceptron.feed_forward(training_x[example])
            y_expected = training_y[example][0]
            error = y_expected - y_predicted
            total_error += error
            perceptron.update_weights(training_x[example], error)
        print("epoch " + str(epoch) + " Total Error " + str(total_error))
        if total_error == 0:
            break
    print("Final Weights : " + str(perceptron.weights))

    "Testing final weights"
    print("\nTesting final weights")
    print('Input [-1, -1] Output ' + str(perceptron.feed_forward([-1, -1])))
    print('Input [-1, +1] Output ' + str(perceptron.feed_forward([-1, +1])))
    print('Input [+1, -1] Output ' + str(perceptron.feed_forward([+1, -1])))
    print('Input [+1, +1] Output ' + str(perceptron.feed_forward([+1, +1])))
