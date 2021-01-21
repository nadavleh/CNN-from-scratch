import mnist
import numpy as np
from covNet import *
import matplotlib.pyplot as plt
import time

np.set_printoptions(edgeitems=100, linewidth=200000)

cnn = covNet(6, 12)

train_images = (mnist.train_images() / 255) - 0.5
train_labels = mnist.train_labels()

test_images = (mnist.test_images() / 255) - 0.5
test_labels = mnist.test_labels()

stats = cnn.train(train_images[:1000], train_labels[:1000], test_images[:100], test_labels[:100], epoch=10, lr = 0.01)

epochs = stats[0]
avg_losses = stats[1]
accuracies = stats[2]



fig = plt.figure()
plt.subplots_adjust(hspace=0.5)

g1 = fig.add_subplot(2, 1, 1, ylabel="Loss", xlabel="Epoch")
g1.plot(epochs, avg_losses, label="Avg loss", color="red")
g1.legend(loc="center")

g2 = fig.add_subplot(2, 1, 2, ylabel="Accuracy", xlabel="Epoch")
g2.plot(epochs, accuracies, label="Accuracy", color="green")
g2.legend(loc="center")

plt.show()
