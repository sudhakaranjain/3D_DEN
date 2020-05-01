import tensorflow as tf
import gzip
import cv2 as cv
import os
import numpy as np

class DEN():

	def __init__(self):
		self.last_label_index = 0    # tracks the label index upto which the model has been trained
		self.params = dict()
		self.k_ex = 5
		self.den_layers = 3
		tf.reset_default_graph()
		self.sess = None
		self.train = []
		self.train_labels = []
		self.test = []
		self.test_labels = []
		self.selected = dict()
		self.l2_mu = 0.001
		self.l1_thr = 0.001
		
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

			for i in range(len(self.train_labels)):
				self.train_labels[i] = 0

		for index in new_label_indices:
			print(" \n Added new category: "+str(label_names[index]))
			l = 1 if task_id != 1 else index
			for data, label in zip(train_data, train_labels):
				if label_names[index] == label:
					self.train.append(data)
					self.train_labels.append(l)

			for data, label in zip(test_data, test_labels):
				if label_names[index] == label:
					self.test.append(data)
					self.test_labels.append(index)

	def destroy_graph(self):
		tf.reset_default_graph()
		self.params = dict()

	def initialize_parameters(self, output_len=2):
		self.sess = tf.Session()
		self.x = tf.placeholder(tf.float32, [None, 60, 60, 3])
		self.y_ = tf.placeholder(tf.float32, [None, output_len])
		# self.keep_prob = tf.placeholder(tf.float32)  			# dropout probability

	def create_variable(self, name=None, shape=None, scope=None, trainable=True):
		with tf.variable_scope(scope, reuse=False):
			w = tf.get_variable(name, shape=shape, trainable=trainable)
			if "task" not in name:
				self.params[w.name] = w
		return w

	def get_variable(self, name=None, scope=None):
		with tf.variable_scope(scope, reuse=True):
			w = tf.get_variable(name)
			if "task" not in name:
				self.params[w.name] = w
		return w

	def restore_params(self, trainable=True, param_values=dict(), retraining=False):
		self.params = dict()
		for scope_name, value in param_values.items():
			scope_name = scope_name.split(':')[0]
			[scope, name] = scope_name.split('/')

			if retraining == False:
				trainable = False
			elif ('l%d' % den_layers in scope) or ("conv" in scope):
				trainable = False
			else:
				trainable = True

			with tf.variable_scope(scope, reuse=False):
				w = tf.get_variable(name, initializer=value, trainable=trainable)
			self.params[w.name] = w

	def get_params(self):
		vdict = dict()
		for scope_name, ref_w in self.params.items():
			vdict[scope_name] = self.sess.run(ref_w)
		return vdict

	def conv2d(self, x, W):
		return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

	def max_pool_2x2(self, x):
		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
							strides=[1, 2, 2, 1], padding='SAME')

	
	def build_model(self, task_id, retraining=False, expansion=False, splitting=False, prediction=False, output_len=2):

		# Note: scope and name values are only given to DEN layers, not for fixed sized layers.
		
		self.initialize_parameters(output_len)

		if task_id == 1:

			#First Convolutional layers
			W_conv1 = self.create_variable(name="w",shape=[5, 5, 3, 32],scope="conv1")
			b_conv1 = self.create_variable(name="b",shape=[32],scope="conv1")

			h_conv1 = tf.nn.relu(self.conv2d(self.x, W_conv1) + b_conv1)
			h_pool1 = self.max_pool_2x2(h_conv1)

			#Second Convolutional Layer
			W_conv2 = self.create_variable(name="w",shape=[5, 5, 32, 64],scope="conv2")
			b_conv2 = self.create_variable(name="b",shape=[64],scope="conv2")

			h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
			h_pool2 = self.max_pool_2x2(h_conv2)
			self.h_pool2_flat = tf.reshape(h_pool2, [-1, 15*15*64])

			#flattened first fc layer
			W_fc1 = self.create_variable(name="w", shape=[15 * 15 * 64, 1024], scope="l1")   # layer-1 outgoing weight matrix
			b_fc1 = self.create_variable(name="b", shape=[1024], scope="l1")

			h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, W_fc1) + b_fc1)

			#second fc layer
			W_fc2 = self.create_variable(name="w", shape=[1024, 128], scope="l2")   # layer-2 outgoing weight matrix
			b_fc2 = self.create_variable(name="b", shape=[128], scope="l2")

			self.h_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2

			# readout fc layer
			self.w_fc = self.create_variable(name="w", shape=[128, output_len], scope="l3")     # layer-3 outgoing weight matrix 
			self.b_fc = self.create_variable(name="b", shape=[output_len], scope="l3")
			y_conv = tf.matmul(self.h_fc2, self.w_fc) + self.b_fc

		else:

			#First Convolutional layers
			W_conv1 = self.get_variable(name="w",scope="conv1")
			b_conv1 = self.get_variable(name="b",scope="conv1")

			h_conv1 = tf.nn.relu(self.conv2d(self.x, W_conv1) + b_conv1)
			h_pool1 = self.max_pool_2x2(h_conv1)

			#Second Convolutional Layer
			W_conv2 = self.get_variable(name="w",scope="conv2")
			b_conv2 = self.get_variable(name="b",scope="conv2")

			h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
			h_pool2 = self.max_pool_2x2(h_conv2)
			self.h_pool2_flat = tf.reshape(h_pool2, [-1, 15*15*64])

			#flattened first fc layer
			W_fc1 = self.get_variable(name="w", scope="l1")   # layer-1 outgoing weight matrix
			b_fc1 = self.get_variable(name="b", scope="l1")

			h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, W_fc1) + b_fc1)

			#second fc layer
			W_fc2 = self.get_variable(name="w", scope="l2")   # layer-2 outgoing weight matrix
			b_fc2 = self.get_variable(name="b", scope="l2")

			self.h_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2

			self.w_fc = self.create_variable(name="w_task_"+str(task_id), shape=[128, output_len], scope="l3", trainable=True)     # layer-3 outgoing weight matrix 
			self.b_fc = self.create_variable(name="b_task_"+str(task_id), shape=[output_len], scope="l3", trainable=True)
			y_conv = tf.matmul(self.h_fc2, self.w_fc) + self.b_fc

		return y_conv

	def optimization(self):

		l2_regular = 0
		train_var = []

		loss = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_, logits=y_conv))

		for var in tf.trainable_variables():
			l2_regular = l2_regular + tf.nn.l2_loss(var)
			train_var.append(var)

		loss = loss + tf.reduce_mean(self.l2_mu * l2_regular)

		opt = tf.train.AdamOptimizer(1e-5)
		grads = opt.compute_gradients(loss, train_var)
		apply_grads = opt.apply_gradients(grads)

		correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(self.y_,1))
		self.acc_train = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		with tf.control_dependencies([apply_grads]):
			self.train_model = tf.no_op()

	def train_task(self, task_id, batch_size, epochs):

		if task_id == 1:
			task_train_labels = tf.one_hot(indices=np.array(self.train_labels), depth=self.last_label_index+1)
		else:
			task_train_labels = np.array(self.train_labels)
			task_train_labels = task_train_labels.reshape((task_train_labels.shape[0],1))
			task_train_labels = tf.convert_to_tensor(task_train_labels)

		dataset = tf.data.Dataset.from_tensor_slices((np.array(self.train), task_train_labels))
		dataset = dataset.shuffle(len(self.train_labels)).repeat().batch(batch_size)
		iterator = dataset.make_one_shot_iterator()
		(x_data , y_data) = iterator.get_next()

		self.sess.run(tf.global_variables_initializer())
		task_train_labels = self.sess.run(task_train_labels)
		count = 0

		print("\n-------------Training new task: %d--------------\n"%task_id)
		for i in range(epochs):

			for j in range(int(len(self.train)/batch_size)):
				x_batch , y_batch = self.sess.run([x_data,y_data])
				self.sess.run(self.train_model, feed_dict={self.x: x_batch, self.y_: y_batch})
			
			train_accuracy = self.acc_train.eval(session=self.sess, feed_dict={self.x: self.train, self.y_: task_train_labels})
			print("Epoch %d, training accuracy %g"%(i+1, train_accuracy))

			if train_accuracy >= 0.99:
				count += 1

				if count > 3:
					print("Best accuracy achieved! \n")
					break

	def predict(self, task_id, output_len=2):

		self.y_ = tf.placeholder(tf.float32, [None, output_len])

		task_test_labels = tf.one_hot(indices=self.test_labels, depth=self.last_label_index+1)
		task_test_labels = self.sess.run(task_test_labels)

		self.w_fc = self.get_variable(name="w", scope="l3")
		self.b_fc = self.get_variable(name="b", scope="l3")

		if task_id != 1:

			t_w = self.get_variable(name="w_task_"+str(task_id), scope="l3")
			t_b = self.get_variable(name="b_task_"+str(task_id), scope="l3")
			
			w_name = self.w_fc.name
			b_name = self.b_fc.name
			
			self.w_fc = tf.concat([self.w_fc, t_w], 1)
			self.b_fc = tf.concat([self.b_fc, t_b], 0)

			self.params[w_name] = self.w_fc
			self.params[b_name] = self.b_fc

		y_final = tf.matmul(self.h_fc2, self.w_fc) + self.b_fc

		correct_prediction = tf.equal(tf.argmax(y_final,1), tf.argmax(self.y_,1))
		self.acc_test = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		test_accuracy = self.acc_test.eval(session=self.sess, feed_dict={self.x: self.test, self.y_: task_test_labels})
		print("Overall accuracy: %g \n"%test_accuracy)


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

	batch_size = 10
	epochs = 10

	den = DEN()

	if os.path.exists(np_train_path):

		print("\n..........loading dataset from numpy files..........\n")

		with gzip.GzipFile(np_train_path, "r") as f:
			train_data = np.load(f)
		with gzip.GzipFile(np_test_path, "r") as f:
			test_data = np.load(f)

		train_labels = np.load(np_train_label_path)
		test_labels = np.load(np_test_label_path)
		label_names = np.load(np_label_name_path)

	else:

		print("\n..........loading dataset from disk..........\n")
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

	task_id = 0
	y_conv = None

	while den.last_label_index != (len(label_names) - 1):     # Loop for adding new tasks (lifelong learning)

		task_id += 1
		den.add_task(task_id, label_names)
		param_values = dict()

		if task_id == 1:
			y_conv = den.build_model(task_id=task_id)
			den.optimization()
			den.train_task(task_id=task_id, batch_size=batch_size, epochs=epochs)
		else:
			param_values = den.get_params()              # Backing-up model weights that were trained upto prev task
			den.destroy_graph()
			den.restore_params(param_values=param_values)

			y_conv = den.build_model(task_id=task_id, output_len=1)
			den.optimization()
			den.train_task(task_id=task_id, batch_size=batch_size, epochs=epochs)

		# ---------Performance-----------
		den.predict(task_id=task_id, output_len=den.last_label_index+1)
		