import numpy as np
from lib import *
from mnist.loader import MNIST

mndata = MNIST('./data')
images, labels = mndata.load_training()
t_images, t_labels = mndata.load_testing()

images = np.array(images)
t_images = np.array(t_images)
labels = np.array(labels)
t_labels = np.array(t_labels)

print_statistics(images, t_images, labels, t_labels)
