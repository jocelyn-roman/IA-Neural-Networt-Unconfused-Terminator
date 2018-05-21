import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
# import os  # in case of file saving


def visualize_image(x, title):
    # Based on: https://www.quora.com/How-can-l-visualize-cifar-10-data-RGB-using-python-matplotlib
    img = x.reshape(28, 28)
    plt.imshow(img, cmap='gray')
    plt.title("Feed backward of " + title)

    # Uncomment this to show the image
    plt.show()

    # Uncomment the following code to save into disk
    '''
    directory = os.path.abspath("output/backward")
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + "/" + title)
    '''


def plot_probability(probability):
    # Based on: https://plot.ly/matplotlib/bar-charts/
    y = probability
    x = range(10)
    width = 1 / 1.5
    plt.bar(x, y, width, color="blue")
    plt.show()


def load_image(file):
    # Receive a file path in string
    image = Image.open(file)
    # Converting to B&W
    image = image.convert('L')
    # Resizing
    image = image.resize((28, 28))
    # Get image as numpy array
    raw_image = np.array(list(image.getdata()))
    raw_image = raw_image.reshape(1, raw_image.shape[0])
    # Close image
    image.close()

    return raw_image