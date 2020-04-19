import tensorflow as tf
import gzip
import cv2 as cv
import os
import numpy as np

class DEN():

	def __init__(self):
		self.last_label_index = 0    # tracks the label index upto which the model has been trained
		self.params = dict()
		self.train = []
		self.train_labels = []
		self.test = []
		self.test_labels = []
		tf.reset_default_graph()


	def extract_data(self, filepath):
		data = []
		data_labels = []
		label_names = dict()
		i = 1
		for label in os.listdir(filepath):
			label_path = os.path.join(filepath, label)
			count = 0
			list = len(os.listdir(label_path))
			for img in os.listdir(label_path):
				image_path = os.path.join(label_path, img)
				image = cv.imread(image_path)
				re_image = cv.resize(image, (60,60))
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


	def add_task(self, task_id, label_names, initial_output=2):

		new_label_indices = []

		if task_id == 1:
			self.train = []
			self.train_labels = []
			self.test = []
			self.test_labels = []
			for i in range(initial_output):   # By default, first task is a binary classification
				new_label_indices.append(i)
			self.last_label_index = i

		else:
			new_label_indices.append(self.last_label_index+1)
			self.last_label_index = self.last_label_index + 1

		for index in new_label_indices:
			print(" \n Added new category: "+str(label_names[index]))
			for data, label in zip(train_data, train_labels):
				if label_names[index] == label:
					self.train.append(data)
					self.train_labels.append(index) 

			for data, label in zip(test_data, test_labels):
				if label_names[index] == label:
					self.test.append(data)
					self.test_labels.append(index)

		return [np.array(self.train), np.array(self.train_labels), 
					np.array(self.test), np.array(self.test_labels)]

	# def weight_variable(self, task_id, shape, scope=None, trainable=True):
	# 	with tf.variable_scope(scope, reuse=True):
	# 		w = tf.get_variable(task_id, shape, trainable=trainable)
	# 		self.params[w.name] = w
			
	# 	return w

	# def bias_variable(self, shape, scope=None, trainable=True):
	# 	if den == False
	# 		initial = tf.constant(0.1, shape=shape)
	# 		return tf.Variable(initial)
	# 	else:


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

	def build_model(self, x, keep_prob, retraining=False, expansion=False, splitting=False, output_len=2):

		#First Convolutional Layer
		W_conv1 = self.weight_variable([5, 5, 3, 32])
		b_conv1 = self.bias_variable([32])

		h_conv1 = tf.nn.relu(self.conv2d(x, W_conv1) + b_conv1)
		h_pool1 = self.max_pool_2x2(h_conv1)


		#Second Convolutional Layer
		W_conv2 = self.weight_variable([5, 5, 32, 64])
		b_conv2 = self.bias_variable([64])

		h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
		h_pool2 = self.max_pool_2x2(h_conv2)


		#flattened first fc layer
		W_fc1 = self.weight_variable([15 * 15 * 64, 1024])
		b_fc1 = self.bias_variable([1024])


		h_pool2_flat = tf.reshape(h_pool2, [-1, 15*15*64])
		h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

		#Dropout Layer
		h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

		#second fc layer
		W_fc2 = self.weight_variable([1024, 256])
		b_fc2 = self.bias_variable([256])

		h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

		# readout fc layer
		W_fc3 = self.weight_variable([256, output_len])
		b_fc3 = self.bias_variable([output_len])

		y_conv = tf.matmul(h_fc2, W_fc3) + b_fc3

		return y_conv

if __name__ == "__main__":

	# train_data_path = '../Fruit_dataset/Train'
	# test_data_path = '../Fruit_dataset/Test'

	# lesser categories for faster loading during testing the code
	train_data_path = '../Fruit_dataset/temp_Train'
	test_data_path = '../Fruit_dataset/temp_Test'

	# For easier disk read operation
	np_train_path = '../Fruit_dataset/numpy_dataset/train.npy.gz'
	np_test_path = '../Fruit_dataset/numpy_dataset/test.npy.gz'
	np_train_label_path = '../Fruit_dataset/numpy_dataset/train_labels.npy'
	np_test_label_path = '../Fruit_dataset/numpy_dataset/test_labels.npy'
	np_label_name_path = '../Fruit_dataset/numpy_dataset/label_names.npy'

	batch_size = 40
	epochs = 30

	den = DEN()

	if os.path.exists(np_train_path):

		print("..........loading dataset from numpy files..........")

		with gzip.GzipFile(np_train_path, "r") as f:
			train_data = np.load(f)
		with gzip.GzipFile(np_test_path, "r") as f:
			test_data = np.load(f)

		train_labels = np.load(np_train_label_path)
		test_labels = np.load(np_test_label_path)
		label_names = np.load(np_label_name_path)

	else:

		print("..........loading dataset from disk..........")
		train_data, train_labels, label_names = den.extract_data(train_data_path)
		test_data, test_labels, _ = den.extract_data(test_data_path)

		os.makedirs(os.path.dirname(np_train_path), exist_ok=True)

		with gzip.GzipFile(np_train_path, "w") as f:
			np.save(f, train_data)
		with gzip.GzipFile(np_test_path, "w") as f:
			np.save(f, test_data)

		np.save(np_train_label_path, train_labels)
		np.save(np_test_label_path, test_labels)
		np.save(np_label_name_path, label_names)


# show image using cv
	# print(train_labels[512])
	# cv.imshow("", train_data[512])
	# cv.waitKey(0)

	x = tf.placeholder(tf.float32, [None, 60, 60, 3])
	y_ = tf.placeholder(tf.float32, [None, len(label_names)])
	keep_prob = tf.placeholder(tf.float32)

	y_conv = den.build_model(x, keep_prob, output_len=len(label_names))
	task_id = 0

	while den.last_label_index != (len(label_names) - 1):

		task_id += 1
		# if task_id == 1:
		# 	y_conv = den.build_model(task_id, x, keep_prob)

		[task_train, new_task_train_labels, task_test, new_task_test_labels] = den.add_task(task_id, label_names)

		task_train_labels = tf.one_hot(indices=new_task_train_labels, depth=len(label_names))
		task_test_labels = tf.one_hot(indices=new_task_test_labels, depth=len(label_names))					

		dataset = tf.data.Dataset.from_tensor_slices((task_train, task_train_labels))
		dataset = dataset.shuffle(len(new_task_train_labels)).repeat().batch(batch_size)
		iterator = dataset.make_one_shot_iterator()
		(x_data , y_data) = iterator.get_next()

		cross_entropy = tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv))
		train_model = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)
		correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		with tf.Session() as sess:

			sess.run(tf.global_variables_initializer())
			task_test_labels = sess.run(task_test_labels)
			task_train_labels = sess.run(task_train_labels)
			count = 0

			print("\n-------------Training new task: %d--------------\n"%task_id)
			for i in range(epochs):

				for j in range(int(len(task_train)/batch_size)):
					x_batch , y_batch = sess.run([x_data,y_data])
					sess.run(train_model, feed_dict={x: x_batch, y_: y_batch, keep_prob: 0.5})
				
				train_accuracy = accuracy.eval(feed_dict={x:task_train, y_: task_train_labels, keep_prob: 1.0})
				print("Epoch %d, training accuracy %g"%(i+1, train_accuracy))

				test_accuracy = accuracy.eval(feed_dict={x: task_test, y_: task_test_labels, keep_prob: 1.0})
				print("test accuracy: %g \n"%test_accuracy)

				if test_accuracy >= 0.99:
					count += 1

					if count > 3:
						print("Best accuracy achieved! \n")
						break
