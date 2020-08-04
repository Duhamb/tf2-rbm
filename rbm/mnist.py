import numpy as np
import matplotlib.pyplot as plt
from rbm import *

(train_data, _), (test_data, _) = tf.keras.datasets.mnist.load_data()
train_data = train_data/np.float32(255)
train_data = np.reshape(train_data, (train_data.shape[0], 784))
test_data = test_data/np.float32(255)
test_data = np.reshape(test_data, (test_data.shape[0], 784))

#Size of inputs is the number of inputs in the training set
input_size = train_data.shape[1]
rbm = RBM(input_size, 200)
err = rbm.train(train_data,50)

plt.plot(err)
plt.xlabel('epochs')
plt.ylabel('cost')
plt.show()

out = rbm.rbm_reconstruct(test_data)

# Plotting original and reconstructed images
row, col = 2, 8
idx = np.random.randint(0, 100, row * col // 2)
f, axarr = plt.subplots(row, col, sharex=True, sharey=True, figsize=(20,4))

for fig, row in zip([test_data,out], axarr):
	for i,ax in zip(idx,row):
		ax.imshow(tf.reshape(fig[i],[28, 28]), cmap='Greys_r')
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)

plt.show()
