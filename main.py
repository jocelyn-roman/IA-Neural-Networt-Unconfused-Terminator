"""
Created on Apr 27, 2018

@author: J&J
"""

from mnist import MNIST
import numpy as np
import neural_network as nn
import utils as utl


def main():
    # Loading MNIST data set
    print("Loading MNIST data set...")
    data = MNIST("./MNIST_data_set")
    training_images, training_labels = data.load_training()

    # Converting to numpy arrays and normalizing images
    print("Preparing data...")
    training_images = np.array(training_images) / 255
    training_labels = np.array(training_labels)

    # Getting dimensions
    first_layer = training_images.shape[1]
    last_layer = training_labels.max() + 1

    # Creating neural network
    print("Initializing neural network...")
    neural_network = nn.OneHiddenLayer(first_layer, 1024, last_layer)

    # Training neural network
    print("Training...")
    neural_network.train(training_images, training_labels, 32, 5)
    neural_network.plot()

    # Saving weights into file
    print("Saving weights")
    neural_network.save("weights.pickle")


def test_data():
    # Loading MNIST data set
    print("Loading MNIST data set...")
    data = MNIST("./MNIST_data_set")
    test_images, test_labels = data.load_testing()

    # Converting to numpy arrays and normalizing images
    print("Preparing data...")
    test_images = np.array(test_images) / 255
    test_labels = np.array(test_labels)

    # Getting dimensions
    first_layer = test_images.shape[1]
    last_layer = test_labels.max() + 1

    # Creating neural network
    print("Initializing neural network...")
    neural_network = nn.OneHiddenLayer(first_layer, 1024, last_layer)

    # Loading weights into neural network
    print("Loading weights...")
    neural_network.load("weights.pickle")

    # Testing the neural network with the test data set
    print("Testing...")
    loss, accuracy = neural_network.test(test_images, test_labels)
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)


def test_custom_numbers():
    # Setting up neural network
    first_layer = 784
    last_layer = 10
    neural_network = nn.OneHiddenLayer(first_layer, 1024, last_layer)
    neural_network.load("weights.pickle")

    print("Testing with a local image")
    image = utl.load_image("Test_data/zero_1.png")
    probability = neural_network.forward(image)
    print(np.argmax(probability))
    utl.plot_probability(probability[0])

    image = utl.load_image("Test_data/uno.png")
    probability = neural_network.forward(image)
    print(np.argmax(probability))
    utl.plot_probability(probability[0])

    image = utl.load_image("Test_data/two_1.png")
    probability = neural_network.forward(image)
    print(np.argmax(probability))
    utl.plot_probability(probability[0])

    image = utl.load_image("Test_data/three_1.png")
    probability = neural_network.forward(image)
    print(np.argmax(probability))
    utl.plot_probability(probability[0])

    image = utl.load_image("Test_data/cuatro.png")
    probability = neural_network.forward(image)
    print(np.argmax(probability))
    utl.plot_probability(probability[0])

    image = utl.load_image("Test_data/five_1.png")
    probability = neural_network.forward(image)
    print(np.argmax(probability))
    utl.plot_probability(probability[0])

    image = utl.load_image("Test_data/seis.png")
    probability = neural_network.forward(image)
    print(np.argmax(probability))
    utl.plot_probability(probability[0])

    image = utl.load_image("Test_data/siete2.png")
    probability = neural_network.forward(image)
    print(np.argmax(probability))
    utl.plot_probability(probability[0])

    image = utl.load_image("Test_data/eight_1.png")
    probability = neural_network.forward(image)
    print(np.argmax(probability))
    utl.plot_probability(probability[0])

    image = utl.load_image("Test_data/nueve.png")
    probability = neural_network.forward(image)
    print(np.argmax(probability))
    utl.plot_probability(probability[0])


def test_feed_backward():
    # Setting up neural network
    first_layer = 784
    last_layer = 10
    neural_network = nn.OneHiddenLayer(first_layer, 1024, last_layer)
    neural_network.load("weights.pickle")

    result = np.array([[0.01, 0.01, 0.01, 0.91, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]])

    image = neural_network.feed_backward(result) * 255
    utl.visualize_image(image, 1, "")


if __name__ == "__main__":
    main()
    test_data()
    test_custom_numbers()
    test_feed_backward()
