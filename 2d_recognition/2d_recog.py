import tensorflow as tf
import gzip
import cv2 as cv
import os
import numpy as np

class DEN():

	def __init__(self):
		self.last_label_index = 0

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
		label_names = np.unique(data_labels)
		# data_labels = tf.one_hot(indices=data_labels, depth=10)
		return data, data_labels, label_names


	def add_task(self, label_names, binary_output=True):


		new_label_indices = []
		new_train = []
		new_train_labels = []
		new_test = []
		new_test_labels = []

		if binary_output == False:
			new_label_indices.append(label_names[self.last_label_index+1])
			self.last_label_index = self.last_label_index + 1

		else:
			for i in range(no_of_labels):
				new_label_indices.append(label_names[i])
			self.last_label_index = i


		for index in new_label_indices:
			for data, label in zip(train_data, train_labels):
				if label_names[index] == label:
					new_train.append(data)
					new_train_labels.append(label)

			for data, label in zip(test_data, test_labels):
				if label_names[index] == label:
					new_test.append(data)
					new_test_labels.append(label)

		return [np.array(new_train), np.array(new_train_labels), 
					np.array(new_test), np.array(new_test_labels)]

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

	def create_model(self, x, keep_prob, prev_y_conv=None):

		tf.reset_default_graph()

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

		if prev_y_conv == None:
			# readout fc layer
			W_fc3 = self.weight_variable([128, 2])
			b_fc3 = self.bias_variable([2])
		else:

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
	np_label_name_path = '../Fruit_dataset/numpy_dataset/label_names.npy'

	batch_size = 10
	epochs = 50

	den = DEN()

	if os.path.exists(np_train_path):

		print("loading dataset from numpy files.......")
		train_data = np.load(np_train_path)
		train_labels = np.load(np_train_label_path)
		test_data = np.load(np_test_path)
		test_labels = np.load(np_test_label_path)
		label_names = np.load(np_label_name_path)

	else:

		print("loading dataset from image files.......")
		train_data, train_labels, label_names = den.extract_data(train_data_path)
		test_data, test_labels, _ = den.extract_data(test_data_path)

		np.save(np_train_path, train_data)
		np.save(np_train_label_path, train_labels)
		np.save(np_test_path, test_data)
		np.save(np_test_label_path, test_labels)
		np.save(np_label_name_path, label_names)


# show image using cv
	# print(train_labels[512])
	# cv.imshow("", train_data[512])
	# cv.waitKey(0)

	x = tf.placeholder(tf.float32, [None, 50, 50, 3])
	y_ = tf.placeholder(tf.float32, [None, 2])
	keep_prob = tf.placeholder(tf.float32)


	for task_id in range(1, len(label_names)):

		prev_y = y_conv


		if task_id == 1:
			[task_train, task_train_labels, task_test, task_test_labels] = den.add_task(label_names)
			y_conv = den.create_model(task_id, x, keep_prob)

		else:
			task_train , task_labels = den.add_task(label_names, False)
			y_conv = den.create_model(task_id, x, keep_prob, prev_y)			

		dataset = tf.data.Dataset.from_tensor_slices((task_train task_labels))
		dataset = dataset.repeat().batch(batch_size)
		iterator = dataset.make_one_shot_iterator()
		(x_data , y_data) = iterator.get_next()


	with tf.Session() as sess:

		cross_entropy = tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv))
		train_model = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
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
