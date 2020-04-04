import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import gzip
import cv2 as cv
import os
import numpy as np

class DEN():

	# def __init__(self):
	# 	# mnist = input_data.read_data_sets("../mnist/", one_hot=True)
	# 	tf.reset_default_graph()
		 # self.network = keras.Sequential()
		 # self.network.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=(28,28,1)))
		 # self.network.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
		 # self.network.add(keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
		 # self.network.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
		 # self.network.add(keras.layers.Flatten())
		 # self.network.add(keras.layers.Dense(10, activation='softmax'))

	def extract_data(self, filepath):
		data = []
		data_labels = []

		for label in os.listdir(filepath):
			label_path = os.path.join(filepath, label)
			count = 0
			list = len(os.listdir(label_path))
			for img in os.listdir(label_path):
				image_path = os.path.join(label_path, img)
				image = cv.imread(image_path)
				re_image = cv.resize(image, (50,50))
				# grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
				# (thresh, BW) = cv.threshold(grayImage, 127, 255, cv.THRESH_BINARY)
				data.append(re_image)
				data_labels.append(label)
		
		data = np.array(data, dtype="float") / 255.0
		# data = data.reshape(data.shape[0], 50, 50, 3)
		data_labels = np.array(data_labels)
		return [data, data_labels]


	def weight_variable(self, shape):
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial)

	def bias_variable(self, shape):
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial)

	def conv2d(self, x, W):
		return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

	def max_pool_2x2(self, x):
		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
							strides=[1, 2, 2, 1], padding='SAME')

	def create_model(self, x, keep_prob):

		#First Convolutional Layer
		W_conv1 = self.weight_variable([5, 5, 1, 32])
		b_conv1 = self.bias_variable([32])

		h_conv1 = tf.nn.relu(self.conv2d(x, W_conv1) + b_conv1)
		h_pool1 = self.max_pool_2x2(h_conv1)


		#Second Convolutional Layer
		W_conv2 = self.weight_variable([5, 5, 32, 64])
		b_conv2 = self.bias_variable([64])

		h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
		h_pool2 = self.max_pool_2x2(h_conv2)


		#flattened first fc layer
		W_fc1 = self.weight_variable([7 * 7 * 64, 512])
		b_fc1 = self.bias_variable([512])


		h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
		h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

		#Dropout Layer
		h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

		#second fc layer
		W_fc2 = self.weight_variable([512, 128])
		b_fc2 = self.bias_variable([128])

		h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


		# readout fc layer
		W_fc3 = self.weight_variable([128, 10])
		b_fc3 = self.bias_variable([10])

		y_conv = tf.matmul(h_fc2, W_fc3) + b_fc3

		return y_conv

if __name__ == "__main__":

	train_data_path = '../Fruit_dataset/Train'
	test_data_path = '../Fruit_dataset/Test'

	# For easier disk read operation
	np_train_path = '../Fruit_dataset/numpy_dataset/train.npy'
	np_test_path = '../Fruit_dataset/numpy_dataset/test.npy'
	np_train_label_path = '../Fruit_dataset/numpy_dataset/train_labels.npy'
	np_test_label_path = '../Fruit_dataset/numpy_dataset/test_labels.npy'

	batch_size = 64
	epochs = 50

	den = DEN()

	if os.path.exists(np_train_path):

		print("loading dataset from numpy files.......")
		train_data = np.load(np_train_path)
		train_labels = np.load(np_train_label_path)
		test_data = np.load(np_test_path)
		test_labels = np.load(np_test_label_path)

	else:

		print("loading dataset from image files.......")
		[train_data, train_labels] = den.extract_data(train_data_path)
		[test_data, test_labels] = den.extract_data(test_data_path)

		np.save(np_train_path, train_data)
		np.save(np_train_label_path, train_labels)
		np.save(np_test_path, test_data)
		np.save(np_test_label_path, test_labels)


# show image using cv
	# print(train_labels[512])
	# cv.imshow("", train_data[512])
	# cv.waitKey(0)

	x = tf.placeholder(tf.float32, [None, 50, 50, 1])
	y_ = tf.placeholder(tf.float32, [None,10])
	keep_prob = tf.placeholder(tf.float32)

	y_conv = den.create_model(x,keep_prob)

	dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
	dataset = dataset.repeat(epochs).batch(batch_size)
	iterator = dataset.make_one_shot_iterator()
	(x_data , y_data) = iterator.get_next()

	with tf.Session() as sess:

		cross_entropy = tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv))
		train_model = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
		correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		sess.run(tf.global_variables_initializer())
		test_labels = sess.run(test_labels)

		for i in range(epochs):

			for j in range(int(len(train_data)/batch_size)):
					x_batch , y_batch = sess.run([x_data,y_data])
					sess.run(train_model, feed_dict={x: x_batch, y_: y_batch, keep_prob: 0.5})
					train_accuracy = accuracy.eval(feed_dict={x:x_batch, y_: y_batch, keep_prob: 1.0})
					print("step %d, training accuracy %g"%(i, train_accuracy))

			print("-----------------------test accuracy %g---------------------------"%accuracy.eval(feed_dict={
				x: test_data, y_: test_labels, keep_prob: 1.0}))
