import numpy as np
import matplotlib.pyplot as plt
import os
from rbm import *
from data import *

try:
	train_data = np.load('data/train_data.npy')
	print("Backup train_data found and loaded")

except FileNotFoundError:
	print("Backup train_data not found. Generating a new one")

	# Read BVH files and append to train_data array 
	train_data = np.empty((0,96))

	filenames = os.listdir('raw_data')
	print("Filenames: ", filenames)

	for name in filenames:
		print("Opening file: ", name)
		file = open('raw_data/' + name,'r')
		lines = file.readlines()
		for line in lines[187:]:
			array = line.replace(' ', ',')
			array = array.replace('\n', '')
			array = array.split(',')
			array = np.array(array)
			train_data = np.append(train_data, [array], axis = 0)

	train_data = train_data.astype('float32')
	train_data = np.delete(train_data, np.s_[0:6], axis=1)
	f = lambda x: x/360 + 0.5
	train_data = f(train_data)

	np.save('train_data.npy', train_data)

print("Final train_data shape: ", train_data.shape)

#Size of inputs is the number of inputs in the training set
input_size = train_data.shape[1]
rbm = RBM(input_size, 40, 0.001)
err = rbm.train(train_data, 1000)

plt.plot(err)
plt.xlabel('epochs')
plt.ylabel('cost')
plt.show()
