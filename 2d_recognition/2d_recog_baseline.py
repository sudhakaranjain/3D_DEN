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
		self.l1_mu = 0.001
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
		self.train = []
		self.train_labels = []
		
		if task_id == 1:

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


	def destroy_graph(self):
		tf.reset_default_graph()
		self.params = dict()

	def initialize_parameters(self, output_len=2):
		self.sess = tf.Session()
		self.x = tf.placeholder(tf.float32, [None, 60, 60, 3])
		self.y_ = tf.placeholder(tf.float32, [None, output_len])
		# self.keep_prob = tf.placeholder(tf.float32)  			# dropout probability


	def get_variable(self, name=None, shape=None, scope=None, trainable=True):
		with tf.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
			w = tf.get_variable(name, shape=shape, trainable=trainable)
		self.params[w.name] = w
		return w

	def restore_params(self, retraining=False, trainable=True):
		self.params = dict()
		if retraining==True:
			trainable = False
		for scope_name, value in self.param_values.items():
			scope_name = scope_name.split(':')[0]
			[scope, name] = scope_name.split('/')
			if retraining==True and scope=="l3":         # To make sure all layers except last are untrainable during selective retraining
				trainable = True
			with tf.variable_scope(scope):
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

	def perform_selection(task_id, param_values, output_len):

		self.selected = dict()
		
		self.initialize_parameters(output_len)

		self.restore_params(retraining=True, trainable=False)

		W_conv1 = self.get_variable(name="w",scope="conv1")
		b_conv1 = self.get_variable(name="b",scope="conv1")

		h_conv1 = tf.nn.relu(self.conv2d(self.x, W_conv1) + b_conv1)
		h_pool1 = self.max_pool_2x2(h_conv1)

		#Second Convolutional Layer
		W_conv2 = self.get_variable(name="w",scope="conv2")
		b_conv2 = self.get_variable(name="b",scope="conv2")

		h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
		h_pool2 = self.max_pool_2x2(h_conv2)
		h_pool2_flat = tf.reshape(h_pool2, [-1, 15*15*64])

		for layer in range(1, self.den_layers+1):
			w_fc = None
			b_fc = None		

			for t in range(1, task_id):
				
				#Collecting weights of each task from each layer, then concatenating them according to each layer
				w_fc_task = self.get_variable(name="w_"+str(t),scope="l"+str(layer)) 
				b_fc_task = self.get_variable(name="b_"+str(t),scope="l"+str(layer))

				if w_fc!=None:
					w_fc=tf.concat([w_fc, w_fc_task],1)
					b_fc=tf.concat([b_fc, b_fc_task],0)
				else:
					w_fc = w_fc_task
					b_fc = b_fc_task
			
			if layer == 1:
				y_conv = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc) + b_fc)

			elif layer == 3:     # Adding new unit in output layer for the new task
				w_new_out = self.get_variable(name="w_"+str(task_id),shape=[w_fc.get_shape().as_list()[0], 1], scope="l"+str(layer), trainable=True)
				b_new_out = self.get_variable(name="b_"+str(task_id),shape=[1], scope="l"+str(layer), trainable=True)
				self.params[w_new_out.name] = w_new_out
				self.params[b_new_out.name] = b_new_out
				w_fc = tf.concat([w_fc, w_new_out], 1)
				b_fc = tf.concat([b_fc, b_new_out],0)
				y_conv = tf.nn.relu(tf.matmul(y_conv, w_fc) + b_fc)

			else:
				y_conv = tf.nn.relu(tf.matmul(y_conv, w_fc) + b_fc)

		result = self.train_model(task_id, y_conv, batch_size, epochs, selection=True)


	def build_model(self, task_id, retraining=False, expansion=False, splitting=False, output_len=2):

		# Note: scope and name values are only given to DEN layers, not for fixed sized layers.
		
		self.initialize_parameters(output_len)

		if retraining:
			
			self.selected
				

		elif expansion:


			W_conv1 = self.get_variable(name="w",scope="conv1")
			b_conv1 = self.get_variable(name="b",scope="conv1")

			h_conv1 = tf.nn.relu(self.conv2d(self.x, W_conv1) + b_conv1)
			h_pool1 = self.max_pool_2x2(h_conv1)

			#Second Convolutional Layer
			W_conv2 = self.get_variable(name="w",scope="conv2")
			b_conv2 = self.get_variable(name="b",scope="conv2")

			h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
			h_pool2 = self.max_pool_2x2(h_conv2)
			h_pool2_flat = tf.reshape(h_pool2, [-1, 15*15*64])

			for layer in range(1, self.den_layers+1):
				w_fc = None
				b_fc = None

				if layer == 1:
					for t in range(1, task_id):
						if t==1:
							w_fc = self.get_variable(name="w_"+str(t),scope="l"+str(layer))
							b_fc = self.get_variable(name="b_"+str(t),scope="l"+str(layer))
						else:
							w_fc = tf.concat([w_fc, self.get_variable(name="w_"+str(t),scope="l"+str(layer))], 1)
							b_fc = tf.concat([b_fc, self.get_variable(name="b_"+str(t),scope="l"+str(layer))], 0)

					w_expand = self.get_variable(name="w_"+str(task_id),shape=[w_fc.get_shape().as_list()[0], self.k_ex], scope="l"+str(layer))
					b_expand = self.get_variable(name="b_"+str(task_id),shape=[self.k_ex], scope="l"+str(layer))
					self.params[w_expand.name] = w_expand
					self.params[b_expand.name] = b_expand
					w_expanded = tf.concat([w_fc,w_expand],1)
					b_expanded = tf.concat([b_fc,b_expand],0)

				elif layer == 3:

				else:

					for t in range(1, task_id):
						if t==1:
							w_fc = self.get_variable(name="w_"+str(t),scope="l"+str(layer))
							b_fc = self.get_variable(name="b_"+str(t),scope="l"+str(layer))
						else:
							t_prev_dim = w_fc.get_shape().as_list()[0]
							t_next_dim = w_fc.get_shape().as_list()[1]

							dummy_w = tf.get_variable(name="w_n_"+str(t), shape=[self.k_ex, t_next_dim], 
										initializer=tf.constant_initializer(0.0), scope="l"+str(layer), trainable=False)

							w_fc = tf.concat([w_fc,dummy_w],0)

							w_fc = tf.concat([w_fc, self.get_variable(name="w_"+str(t), scope="l"+str(layer))],1)

					prev_dim = w_fc.get_shape().as_list()[0]
					next_dim = w_fc.get_shape().as_list()[1]
					dummy_w = tf.get_variable(name="w_n_"+str(task_id), shape=[self.k_ex, next_dim], 
										initializer=tf.constant_initializer(0.0), scope="l"+str(layer), trainable=False)

					w_fc = tf.concat([w_fc,dummy_w],0)

					w_expand = self.get_variable(name="w_"+str(task_id), shape=[prev_dim + self.k_ex, self.k_ex], scope="l"+str(layer))
					b_expand = self.get_variable(name="b_"+str(task_id), shape=[self.k_ex], scope="l"+str(layer))
					w_expanded = tf.concat([w_fc,w_expand],1)
					self.params[w_expand.name] = w_expand
					self.params[b_expand.name] = b_expand
					
		elif splitting:

		else:

			#First Convolutional layers
			W_conv1 = self.get_variable(name="w",shape=[5, 5, 3, 32],scope="conv1")
			b_conv1 = self.get_variable(name="b",shape=[32],scope="conv1")

			h_conv1 = tf.nn.relu(self.conv2d(self.x, W_conv1) + b_conv1)
			h_pool1 = self.max_pool_2x2(h_conv1)

			#Second Convolutional Layer
			W_conv2 = self.get_variable(name="w",shape=[5, 5, 32, 64],scope="conv2")
			b_conv2 = self.get_variable(name="b",shape=[64],scope="conv2")

			h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
			h_pool2 = self.max_pool_2x2(h_conv2)
			h_pool2_flat = tf.reshape(h_pool2, [-1, 15*15*64])

			#flattened first fc layer
			W_fc1 = self.get_variable(name="w_"+str(task_id), shape=[15 * 15 * 64, 1024], scope="l1")   # layer-1 outgoing weight matrix
			b_fc1 = self.get_variable(name="b_"+str(task_id), shape=[1024], scope="l1")

			h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

			#Dropout Layer
			# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

			#second fc layer
			W_fc2 = self.get_variable(name="w_"+str(task_id), shape=[1024, 128], scope="l2")   # layer-2 outgoing weight matrix
			b_fc2 = self.get_variable(name="b_"+str(task_id), shape=[128], scope="l2")

			h_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2

			# readout fc layer
			W_fc3 = self.get_variable(name="w_"+str(task_id), shape=[128, output_len], scope="l3")     # layer-3 outgoing weight matrix 
			b_fc3 = self.get_variable(name="b_"+str(task_id), shape=[output_len], scope="l3")

			y_conv = tf.matmul(h_fc2, W_fc3) + b_fc3

			return y_conv

	def train_task(self, task_id, y_conv, batch_size, epochs, selection=False):

		l1_regular = 0
		train_var = []
		l1_op_list = []

		if task_id == 1 or selection == True:
			loss = tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_, logits=y_conv))
			
			for var in tf.trainable_variables():
				if "conv" not in var.name:
					l1_regular = l1_regular + tf.nn.l1_loss(var)
				train_var.append(var)
			loss = tf.reduce_mean(loss + l1_mu * l1_regular)

		else:

			loss = tf.reduce_mean(
				tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_, logits=y_conv))
			train_model = tf.train.AdamOptimizer(1e-5).minimize(loss)
			correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(self.y_,1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		opt = tf.train.AdamOptimizer(1e-5)
		grads = opt.compute_gradients(loss, train_var)
		apply_grads = opt.apply_gradients(grads)

		correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(self.y_,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		with tf.control_dependencies([apply_grads]):
			for var in train_var:
				if "conv" not in var.name: 		# Creating sparse in the network but not involving conv layers
					th_t = tf.fill(tf.shape(var), tf.convert_to_tensor(self.l1_thr))
					zero_t = tf.zeros(tf.shape(var))
					var_temp = var - (th_t * tf.sign(var))
					# [pseudo]:  if |var| < th_t: var = [0];  else: var = var_temp
					l1_op = var.assign(tf.where(tf.less(tf.abs(var), th_t), zero_t, var_temp))
					l1_op_list.append(l1_op)

		with tf.control_dependencies(l1_op_list):
			train_model = tf.no_op()

		task_train_labels = tf.one_hot(indices=self.train_labels, depth=self.last_label_index+1)
		task_test_labels = tf.one_hot(indices=self.test_labels, depth=len(label_names))                 

		dataset = tf.data.Dataset.from_tensor_slices((self.train, task_train_labels))
		dataset = dataset.shuffle(len(task_train_labels)).repeat().batch(batch_size)
		iterator = dataset.make_one_shot_iterator()
		(x_data , y_data) = iterator.get_next()


		self.sess.run(tf.global_variables_initializer())
		task_test_labels = self.sess.run(task_test_labels)
		task_train_labels = self.sess.run(task_train_labels)
		count = 0

		print("\n-------------Training new task: %d--------------\n"%task_id)
		for i in range(epochs):

			for j in range(int(len(self.train)/batch_size)):
				x_batch , y_batch = self.sess.run([x_data,y_data])
				self.sess.run(train_model, feed_dict={self.x: x_batch, self.y_: y_batch})
			
			train_accuracy = accuracy.eval(feed_dict={self.x: self.train, self.y_: task_train_labels})
			print("Epoch %d, training accuracy %g"%(i+1, train_accuracy))

			test_accuracy = accuracy.eval(feed_dict={self.x: self.test, self.y_: task_test_labels})
			print("test accuracy: %g \n"%test_accuracy)

			if test_accuracy >= 0.99:
				count += 1

				if count > 3:
					print("Best accuracy achieved! \n")
					break



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
	epochs = 50

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

	y_conv = None

	while den.last_label_index != (len(label_names) - 1):     # Loop for adding new tasks (lifelong learning)

		task_id += 1
		existing_model = y_conv
		den.add_task(task_id, label_names)
		param_values = dict()

		if task_id == 1:
			y_conv = den.build_model(task_id)
			result = den.train_task(task_id, y_conv, batch_size, epochs)
		else:
			param_values = den.get_params()              # Backing-up model weights that were trained upto prev task

			print("--------Performing Selective Retraining---------")
			
			den.destroy_graph()
			den.perform_selection(task_id=task_id, param_values=param_values, output_len=den.last_label_index+1)
			y_conv = den.build_model(task_id, retraining=True, output_len=den.last_label_index+1)
			result = den.train_task(task_id, y_conv, batch_size, epochs)

			# if result == False:
			# 	print("--------Performing Network Expansion---------")
			# 	den.destroy_graph()
			# 	den.restore_params(param_values)        # Restoring model weights that were trained upto prev task
			# 	y_conv = den.build_model(task_id, expansion=True, output_len=den.last_label_index+1)
			# 	result = den.train_task(task_id, y_conv, batch_size, epochs)

			# 	if result == False:
			# 		print("--------Performing Network Splitting---------")
			# 		den.destroy_graph()
			# 		den.restore_params(param_values)        # Restoring model weights that were trained upto prev task
			# 		y_conv = den.build_model(task_id, splitting=True, output_len=den.last_label_index+1)
			# 		result = den.train_task(task_id, y_conv, batch_size, epochs)
			# 		print("\n Overall accuracy after learning task %d: %g"%(task_id, result))