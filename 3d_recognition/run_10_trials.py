#!/usr/bin/env python
# coding: utf-8

# In[1]:


#To increase cell width of ipynb
# from IPython.core.display import display, HTML
# display(HTML("<style>.container { width:100% !important; }</style>"))


# In[4]:


import tensorflow as tf
from collections import defaultdict
import gzip
import cv2 as cv
import os
import numpy as np
import random
import timeit


# In[2]:


class DEN():

	def __init__(self):
		self.last_label_index = 0    # tracks the label index upto which the model has been trained
		self.params = dict()
		self.k_ex = 5
		self.den_layers = 3
		self.conv_layers = 2
		tf.reset_default_graph()
		self.sess = None
		self.train = []
		self.train_labels = []
		self.test = []
		self.test_labels = []
		self.batch_size = 5
		self.epochs = 50
		self.early_stop = 5
		self.selected = dict()
		self.lr = 1e-3
		self.l2_mu = 0.001
		self.lamba_regular = 0.7
		self.l1_thr = 0.00001
#         self.l1_thr = 0.00001
		self.loss_thr = 0.01

	def extract_data(self):
		train = []
		test = []
		train_labels = []
		test_labels = []
		label_names = []
		
		# For easier disk read operation
#       VGG16 
		# np_train_path = '/data/s3558487/my_thesis/extracted_features/new_VGG16/train.npy.gz'
		# np_test_path = '/data/s3558487/my_thesis/extracted_features/new_VGG16/test.npy.gz'
		# np_train_label_path = '/data/s3558487/my_thesis/extracted_features/new_VGG16/train_labels.npy'
		# np_test_label_path = '/data/s3558487/my_thesis/extracted_features/new_VGG16/test_labels.npy'
		# np_label_names_path = '/data/s3558487/my_thesis/extracted_features/new_VGG16/label_names.npy'
		

#       MobileNetv2
		np_train_path = '/data/s3558487/my_thesis/extracted_features/MobileNetV2/train.npy.gz'
		np_test_path = '/data/s3558487/my_thesis/extracted_features/MobileNetV2/test.npy.gz'
		np_train_label_path = '/data/s3558487/my_thesis/extracted_features/MobileNetV2/train_labels.npy'
		np_test_label_path = '/data/s3558487/my_thesis/extracted_features/MobileNetV2/test_labels.npy'
		np_label_names_path = '/data/s3558487/my_thesis/extracted_features/MobileNetV2/label_names.npy'

#       Custom MobileNetv2
		# np_train_path = '/data/s3558487/my_thesis/extracted_features/custom_MobileNetV2/train.npy.gz'
		# np_test_path = '/data/s3558487/my_thesis/extracted_features/custom_MobileNetV2/test.npy.gz'
		# np_train_label_path = '/data/s3558487/my_thesis/extracted_features/custom_MobileNetV2/train_labels.npy'
		# np_test_label_path = '/data/s3558487/my_thesis/extracted_features/custom_MobileNetV2/test_labels.npy'
		# np_label_names_path = '/data/s3558487/my_thesis/extracted_features/custom_MobileNetV2/label_names.npy'
		
		print("\n..........loading dataset from numpy files..........\n")

		with gzip.GzipFile(np_train_path, "r") as f:
			train = np.load(f)
		with gzip.GzipFile(np_test_path, "r") as f:
			test = np.load(f)

		train_labels = np.load(np_train_label_path)
		test_labels = np.load(np_test_label_path)
		label_names = np.load(np_label_names_path)

		return train, train_labels, test, test_labels, label_names


	def add_task(self, task_id, label_names, initial_output=2):

		new_label_indices = []
		self.train = []
		self.train_labels = []
		self.train_ex_labels = []      
 
		if task_id == 1:
			self.new_train = []
			self.new_train_labels = []
			self.new_train_ex_labels = []
			self.total = []
			self.total_ex_labels = []
			self.test = []
			self.test_labels = []
			for i in range(initial_output):   # By default, first task is a binary classification
				new_label_indices.append(i)
			self.last_label_index = i

		else:
			new_label_indices.append(self.last_label_index+1)
			self.last_label_index = self.last_label_index + 1
			
			#saving the old training data
			self.total = self.total + self.new_train
			self.total_ex_labels = self.total_ex_labels + self.new_train_ex_labels
			self.new_train = []
			self.new_train_labels = []
			self.new_train_ex_labels = []

		for index in new_label_indices:
			print(" \n Added new category: "+str(label_names[index]))
			l = 1 if task_id != 1 else index
			for data, label in zip(train_data, train_labels):
				if label_names[index] == label:
					self.new_train.append(data)
					self.new_train_labels.append(l)
					self.new_train_ex_labels.append(index)
				# if len(self.new_train_labels) >= 300:
				#     break

			for data, label in zip(test_data, test_labels):
				if label_names[index] == label:
					self.test.append(data)
					self.test_labels.append(index)
					
		# ------------------ Random sampling old training data ------------------
#         if task_id != 1:
#             sampled_indices = random.sample(range(len(self.total)), len(self.new_train))
#             for k in sampled_indices:
#                 self.train.append(self.total[k])
#                 self.train_labels.append(0)
#                 self.train_ex_labels.append(self.total_ex_labels[k])
				
#             self.train = self.train + self.new_train
#             self.train_labels = self.train_labels + self.new_train_labels
#             self.train_ex_labels = self.train_ex_labels + self.new_train_ex_labels
#         else:
#             self.train = self.new_train
#             self.train_labels = self.new_train_labels
#             self.train_ex_labels = self.new_train_ex_labels
		
		# ------------------- Memory Replay of old training data ----------------------
		if task_id != 1:
			sample_per_label = int(len(self.new_train) / self.last_label_index)
			if sample_per_label < 20:
				sample_per_label = 20
				
			for index in range(self.last_label_index):
				label_indices = []
				for i, m in enumerate(self.total_ex_labels):
					if m == index:
						label_indices.append(i)
				try:
					sampled_label_indices = random.sample(label_indices, k=sample_per_label)
				except:
					print("\n Less samples; so taking them all")
					sampled_label_indices = label_indices
				for k in sampled_label_indices:
					self.train.append(self.total[k])
					self.train_labels.append(0)
					self.train_ex_labels.append(self.total_ex_labels[k])
					
			self.train = self.train + self.new_train
			self.train_labels = self.train_labels + self.new_train_labels
			self.train_ex_labels = self.train_ex_labels + self.new_train_ex_labels
					
		else:
			self.train = self.new_train
			self.train_labels = self.new_train_labels
			self.train_ex_labels = self.new_train_ex_labels
			

	def destroy_graph(self):
		tf.reset_default_graph()
		self.params = dict()

	def initialize_parameters(self, output_len=2):
		self.sess = tf.Session()
		# VGG16
		# self.x = tf.placeholder(tf.float32, [None, 4 * 4 * 512])
		# MobileNetV2
		self.x = tf.placeholder(tf.float32, [None, 1280])
		self.y_ = tf.placeholder(tf.float32, [None, output_len])
		# self.keep_prob = tf.placeholder(tf.float32)          # dropout probability

	def create_variable(self, name=None, shape=None, scope=None, trainable=True):
		with tf.variable_scope(scope, reuse=False):
			w = tf.get_variable(name, shape=shape, 
#                                 initializer=tf.random_normal_initializer(mean=0, stddev=1, seed=3),
								trainable=trainable)
			if "ex" not in name:
				self.params[w.name] = w
		return w

	def get_variable(self, name=None, scope=None):
		with tf.variable_scope(scope, reuse=True):
			w = tf.get_variable(name)
			if "ex" not in name:
				self.params[w.name] = w
		return w

	def restore_params(self, task_id=None, trainable=True, param_values=dict()):
		self.params = dict()
		self.prev_W = dict()
		for scope_name, value in param_values.items():
			self.prev_W[scope_name] = value
			scope_name = scope_name.split(':')[0]
			[scope, name] = scope_name.split('/')

#             if task_id != None:
#                 if ('l%d/w_%d' % (self.den_layers,task_id) in scope_name) or ('l%d/b_%d' % (self.den_layers,task_id) in scope_name):
#                     trainable = True
#                 else:
#                     trainable = False

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


	def build_model(self, task_id, expansion=False, output_len=2):

		# Note: scope and name values are only given to DEN layers, not for fixed sized layers.

		self.initialize_parameters(output_len)

		if task_id == 1:

			#flattened first fc layer
			W_fc1 = self.create_variable(name="w", shape=[1280, 1024], scope="l1")   # layer-1 outgoing weight matrix
			b_fc1 = self.create_variable(name="b", shape=[1024], scope="l1")

			h_fc1 = tf.nn.relu(tf.matmul(self.x, W_fc1) + b_fc1)

			#second fc layer
			W_fc2 = self.create_variable(name="w", shape=[1024, 256], scope="l2")   # layer-2 outgoing weight matrix
			b_fc2 = self.create_variable(name="b", shape=[256], scope="l2")

			self.h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

			# readout fc layer
			self.w_fc = self.create_variable(name="w", shape=[256, output_len], scope="l3")     # layer-3 outgoing weight matrix 
			self.b_fc = self.create_variable(name="b", shape=[output_len], scope="l3")
			y_conv = tf.matmul(self.h_fc2, self.w_fc) + self.b_fc
			
		elif expansion:
			
			# fc layer expansion
			for layer in range(1, self.den_layers+1):

				if layer == 1:
					w_fc1 = self.get_variable(name="w", scope="l%d"%layer)
					b_fc1 = self.get_variable(name="b", scope="l%d"%layer)

					w_expand = self.create_variable(name="w_ex_"+str(task_id),shape=[w_fc1.get_shape().as_list()[0], self.k_ex], scope="l"+str(layer))
					b_expand = self.create_variable(name="b_ex_"+str(task_id),shape=[self.k_ex], scope="l%d"%layer)
					w_expanded = tf.concat([w_fc1,w_expand],1)
					b_expanded = tf.concat([b_fc1,b_expand],0)
					self.params[w_fc1.name] = w_expanded
					self.params[b_fc1.name] = b_expanded
					h_fc1 = tf.nn.relu(tf.matmul(self.x, w_expanded) + b_expanded)

				elif layer == self.den_layers:
					#weight matrix of current task output
					w_fc3 = self.get_variable(name="w",scope="l%d"%layer)
					b_fc3 = self.get_variable(name="b",scope="l%d"%layer)
					next_dim = w_fc3.get_shape().as_list()[1]
					
					w_expand = self.create_variable(name="w_ex_"+str(task_id), shape=[self.k_ex, next_dim], scope="l%d"%layer)
					w_expanded = tf.concat([w_fc3, w_expand], 0)
					
					self.params[w_fc3.name] = w_expanded
					
					y_conv = tf.matmul(self.h_fc2, w_expanded) + b_fc3

				else:
					w_fc2 = self.get_variable(name="w",scope="l%d"%layer)
					b_fc2 = self.get_variable(name="b",scope="l%d"%layer)

					prev_dim = w_fc2.get_shape().as_list()[0]
					next_dim = w_fc2.get_shape().as_list()[1]
					
					# Dummy nodes for prev hidden nodes
					dummy_w = tf.get_variable(name="dummy_t%d_l%d" %(task_id,layer), shape=[self.k_ex, next_dim], 
								initializer=tf.constant_initializer(0.0), trainable=False)

					w_expand = self.create_variable(name="w_ex_"+str(task_id), shape=[prev_dim + self.k_ex, self.k_ex], scope="l%d"%layer)
					b_expand = self.create_variable(name="b_ex_"+str(task_id), shape=[self.k_ex], scope="l%d"%layer)
					
					w_fc2_dummy = tf.concat([w_fc2, dummy_w],0)
					
					w_expanded = tf.concat([w_fc2_dummy, w_expand], 1)
					b_expanded = tf.concat([b_fc2, b_expand], 0)
					
					self.params[w_fc2.name] = w_expanded
					self.params[b_fc2.name] = b_expanded
					self.h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_expanded) + b_expanded)
					
		else:

			#flattened first fc layer
			W_fc1 = self.get_variable(name="w", scope="l1")   # layer-1 outgoing weight matrix
			b_fc1 = self.get_variable(name="b", scope="l1")

			h_fc1 = tf.nn.relu(tf.matmul(self.x, W_fc1) + b_fc1)

			#second fc layer
			W_fc2 = self.get_variable(name="w", scope="l2")   # layer-2 outgoing weight matrix
			b_fc2 = self.get_variable(name="b", scope="l2")

			self.h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

			self.w_fc = self.create_variable(name="w_"+str(task_id), shape=[self.h_fc2.shape[1], output_len], scope="l3", trainable=True)     # layer-3 outgoing weight matrix 
			self.b_fc = self.create_variable(name="b_"+str(task_id), shape=[output_len], scope="l3", trainable=True)
			y_conv = tf.matmul(self.h_fc2, self.w_fc) + self.b_fc

#         y_conv = tf.nn.sigmoid(y_conv)
		return y_conv
	
	def perform_selection(self, task_id, values_dict):      # Breadth first search for selecting non-zero units
		
		all_indices = defaultdict(list)     # to store indices of nonzero units
		selected_params = dict()            # to store values of nonzero units
		selected_prev_params = dict()
		
#         for scope, value in values_dict.items():    # Storing conv layers in selected parameters 
#             if "conv" in scope:
#                 selected_params[scope] = value
			
		for i in reversed(range(1,self.den_layers+1)):
			if i == self.den_layers:
				w = values_dict['l%d/w:0' % i]
				b = values_dict['l%d/b:0' % i]
				w_t = values_dict['l%d/w_%d:0' %(i,task_id)]
				b_t = values_dict['l%d/b_%d:0' %(i,task_id)]
				w = np.concatenate((w, w_t), 1)
				b = np.concatenate((b, b_t), 0)
				for j in range(w.shape[0]):
					if w[j,-1] != 0.0:
						all_indices['l%d' % i].append(j)
				# np.ix_(): fancy indexing, index with arrays of integers
				# Select non-zero weights between the last hidden layer and the output layer
				selected_params['l%d/w_%d:0' % (i, task_id)] = \
								   w[np.ix_(all_indices['l%d' % i], list(range(w.shape[1])))]
				selected_params['l%d/b_%d:0' % (i, task_id)] = b
			else:
				w = values_dict['l%d/w:0' % i]
				b = values_dict['l%d/b:0' % i]
				top_indices = all_indices['l%d' % (i + 1)]
				print("Layer %d: Selecting %d neurons" %(i+1, len(top_indices)))
				for j in range(w.shape[0]):
					if np.count_nonzero(w[j, top_indices]) != 0 or i == 1:
						all_indices['l%d' % i].append(j)
				
				# non-zero weights between the layer i and the layer i+1
				sub_weight = w[np.ix_(all_indices['l%d' % i], top_indices)]
				sub_biases = b[all_indices['l%d' % (i + 1)]]
				selected_params['l%d/w:0' % i] = sub_weight
				selected_params['l%d/b:0' % i] = sub_biases
				
				# prev_W : to avoid drastic change in value of weights (Regularization)
				selected_prev_params['l%d/w:0' % i] = \
								  self.prev_W['l%d/w:0' % i][np.ix_(all_indices['l%d' % i], top_indices)]
				selected_prev_params['l%d/b:0' % i] = \
								   self.prev_W['l%d/b:0' % i][all_indices['l%d' % (i + 1)]]

#         for keys, value in selected_params.items():
#             print(keys)
#             print(value)
				
		return [selected_params, selected_prev_params, all_indices]
		
	def build_SR(self, task_id, selected, output_len):    # creating selective retraining model
		
		self.initialize_parameters(output_len)
		h = self.x
		
		for i in range(1, self.den_layers):
			with tf.variable_scope('l%d' % i):
				w = tf.get_variable('w', initializer=selected['l%d/w:0' % i], trainable=True)
				b = tf.get_variable('b', initializer=selected['l%d/b:0' % i], trainable=True)
			h = tf.nn.relu(tf.matmul(h, w) + b)
			
		# last layer
		with tf.variable_scope('l%d' % self.den_layers):
			w = tf.get_variable('w_%d' % task_id,
								initializer=selected['l%d/w_%d:0' % (self.den_layers, task_id)], trainable=True)
			b = tf.get_variable('b_%d' % task_id,
								initializer=selected['l%d/b_%d:0' % (self.den_layers, task_id)], trainable=True)

		y_conv = tf.matmul(h, w) + b
#         y_conv = tf.nn.sigmoid(y_conv)
		return y_conv

	def optimization(self, prev_W=None):

		l2_regular = 0
		train_var = []
		regular_terms = []
		
		self.loss = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=y_conv))

#         for var in tf.trainable_variables():
#             l2_regular = l2_regular + tf.nn.l1_loss(var)
#             train_var.append(var)
#         print(len(train_var))

		l1_var = [var for var in tf.trainable_variables()]
		regularizer = tf.contrib.layers.l1_regularizer(self.l2_mu)
		reg_term = tf.contrib.layers.apply_regularization(regularizer, l1_var)
		self.loss = self.loss + reg_term
#         self.loss = self.loss + tf.reduce_mean(self.l2_mu * l2_regular)

		if prev_W != None:
			for var in l1_var:
				if var.name in prev_W.keys():
					prev_w = prev_W[var.name]
					regular_terms.append(tf.nn.l2_loss(var - prev_w))
			self.loss = self.loss + self.lamba_regular * tf.reduce_mean(regular_terms)
		
		opt = tf.train.AdamOptimizer(self.lr)
		grads = opt.compute_gradients(self.loss, l1_var)
		apply_grads = opt.apply_gradients(grads)

		correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(self.y_,1))
		self.acc_train = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		
#         l1_var = [var for var in tf.trainable_variables()]
		l1_op_list = []
		with tf.control_dependencies([apply_grads]):  # exec apply_grads first
			for var in tf.trainable_variables():
				th_t = tf.fill(tf.shape(var), tf.convert_to_tensor(self.l1_thr))
				zero_t = tf.zeros(tf.shape(var))
#                 var_temp = var - (th_t * tf.sign(var))
				# [pseudo]:  if |var| < th_t: var = [0];  else: var = var_temp
				l1_op = var.assign(tf.where(tf.less(tf.abs(var), th_t), zero_t, var))
				l1_op_list.append(l1_op)
		
		with tf.control_dependencies(l1_op_list):
			self.train_model = tf.no_op()

	def train_task(self, task_id, batch_size, epochs, retraining=False, expansion=False):

		if task_id == 1:
			task_train_labels = tf.one_hot(indices=np.array(self.train_labels), depth=self.last_label_index+1)
		elif retraining == True or expansion == True:
			task_train_labels = tf.one_hot(indices=np.array(self.train_ex_labels), depth=self.last_label_index+1)
		else:
			task_train_labels = np.array(self.train_labels)
			task_train_labels = task_train_labels.reshape((task_train_labels.shape[0],1))
			task_train_labels = tf.convert_to_tensor(task_train_labels)

		dataset = tf.data.Dataset.from_tensor_slices((np.array(self.train), task_train_labels))
		dataset = dataset.shuffle(len(self.train_ex_labels)).repeat().batch(batch_size)
		iterator = dataset.make_one_shot_iterator()
		(x_data , y_data) = iterator.get_next()

		self.sess.run(tf.global_variables_initializer())
		task_train_labels = self.sess.run(task_train_labels)
		count = 0

#         for scope, ref in self.params.items():
#             if "l2" in scope:
#                 print(self.sess.run(ref))

		for i in range(epochs):

			for j in range(int(len(self.train)/batch_size)):
				x_batch , y_batch = self.sess.run([x_data,y_data])
				_, loss = self.sess.run([self.train_model, self.loss], feed_dict={self.x: x_batch, self.y_: y_batch})

			train_accuracy = self.acc_train.eval(session=self.sess, feed_dict={self.x: self.train, self.y_: task_train_labels})
			print("Epoch %d, training accuracy %g"%(i+1, train_accuracy))

			if train_accuracy == 1:
				count += 1

				if count > 9:
					print("Best accuracy achieved! \n")
					break
		
#         for scope, ref in self.params.items():
#             if "l2" in scope:
#                 print(self.sess.run(ref))
		
		return loss
	
	def predict(self, task_id, output_len=2):

		self.initialize_parameters(output_len)

		task_test_labels = tf.one_hot(indices=self.test_labels, depth=self.last_label_index+1)
		task_test_labels = self.sess.run(task_test_labels)
		
		#flattened first fc layer
		W_fc1 = self.get_variable(name="w", scope="l1")   # layer-1 outgoing weight matrix
		b_fc1 = self.get_variable(name="b", scope="l1")

		h_fc1 = tf.nn.relu(tf.matmul(self.x, W_fc1) + b_fc1)

		#second fc layer
		W_fc2 = self.get_variable(name="w", scope="l2")   # layer-2 outgoing weight matrix
		b_fc2 = self.get_variable(name="b", scope="l2")

		self.h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

		self.w_fc = self.get_variable(name="w", scope="l3")
		self.b_fc = self.get_variable(name="b", scope="l3")

		y_final = tf.matmul(self.h_fc2, self.w_fc) + self.b_fc
		y_final = tf.nn.softmax(y_final)
		
		self.sess.run(tf.global_variables_initializer())

		correct_prediction = tf.equal(tf.argmax(y_final,1), tf.argmax(self.y_,1))
		self.acc_test = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		test_accuracy = self.acc_test.eval(session=self.sess, feed_dict={self.x: self.test, self.y_: task_test_labels})
		return test_accuracy


# In[3]:


if __name__ == "__main__":
	
	# np_all_accuracies_path = '/data/s3558487/my_thesis/extracted_features/VGG16/all_accuracies.npy'
	np_all_accuracies_path = '/data/s3558487/my_thesis/extracted_features/MobileNetV2/all_accuracies.npy'
	# np_all_time_path = '/data/s3558487/my_thesis/extracted_features/VGG16/all_time.npy'
	np_all_time_path = '/data/s3558487/my_thesis/extracted_features/MobileNetV2/all_time.npy'
	np_s_all_path = '/data/s3558487/my_thesis/extracted_features/MobileNetV2/s_all.npy'
	# np_s_all_path = '/data/s3558487/my_thesis/extracted_features/VGG16/s_all.npy'
	# np_all_accuracies_path = '/data/s3558487/my_thesis/extracted_features/custom_MobileNetV2/all_accuracies.npy'
	# np_all_time_path = '/data/s3558487/my_thesis/extracted_features/custom_MobileNetV2/all_time.npy'
	# np_s_all_path = '/data/s3558487/my_thesis/extracted_features/custom_MobileNetV2/s_all.npy'
	
	all_accuracies = []
	all_time = []
	s_all = []
	
	for t in range(10):
		
		print("\n------------------------------------- Trial: %d --------------------------------------" %(t+1))
		
		den = DEN()
		temp = []   # To store accuracies for every task per trial
		time = []   # To store training time for every task per trial
		s = []      # To store DEN layer sizes for every task per trial
		train_data, train_labels, test_data, test_labels, label_names = den.extract_data()
		np.random.shuffle(label_names)
	
		# show image using cv
		# print(train_labels[512])
		# cv.imshow("", train_data[512])
		# cv.waitKey(0)
	
		tf.reset_default_graph()
		task_id = 0
		y_conv = None
	
		while den.last_label_index != (len(label_names) - 1):     # Loop for adding new tasks (lifelong learning)
	
			task_id += 1
			den.add_task(task_id, label_names)
			param_values = dict()
			selected = dict()
			
			start = timeit.default_timer()
			
			print("-------------Training new task: %d--------------"%task_id)
			if task_id == 1:
				y_conv = den.build_model(task_id=task_id)
				den.optimization()
				_ = den.train_task(task_id=task_id, batch_size=den.batch_size, epochs=den.epochs)
				params = den.get_params()
	
				den.destroy_graph()
				den.restore_params(trainable=False, param_values=params)  # Freezes all learned weights making them non-trainable
	
				# ---------Performance-----------
				test_accuracy = den.predict(task_id=task_id, output_len=den.last_label_index+1)
				
			else:
				
				print("-----------Started Selective Retraining-------------")
				
				#------------------Selection-------------------
				print("---- Selecting nodes ----")
				y_conv = den.build_model(task_id=task_id, output_len=1)
				den.sess.run(tf.global_variables_initializer())
				params = den.get_params()
				den.optimization()
				
				_ = den.train_task(task_id=task_id, batch_size=den.batch_size, epochs=den.early_stop)
				params = den.get_params()
		   
				[selected, selected_prev, all_indices] = den.perform_selection(task_id=task_id, values_dict=params)
				den.destroy_graph()
	
				#------------------Retraining-------------------
				print("\n---- Retraining selected nodes ----")
				y_conv = den.build_SR(task_id=task_id, selected=selected, output_len=den.last_label_index+1) 
				den.optimization(prev_W=selected_prev) 
				loss = den.train_task(task_id=task_id, batch_size=den.batch_size, epochs=den.epochs, retraining=True)
	
				print("Loss: %f"%loss)
	
				#--------------Performing Union----------------
				_vars = [(var.name, den.sess.run(var)) for var in tf.trainable_variables() if 'l' in var.name]
	
				for item in _vars:
					key, values = item
					selected[key] = values
	
				for i in reversed(range(1, den.den_layers+1)):
					if i == den.den_layers:
						temp_weight = np.concatenate((params['l%d/w:0' % i], params['l%d/w_%d:0' % (i, task_id)]), axis=1)
						temp_weight[np.ix_(all_indices['l%d' % i], list(range(den.last_label_index+1)))] =                         selected['l%d/w_%d:0' % (i, task_id)]
						# Updating output matrix structure
						params['l%d/w:0' % (i)] = temp_weight.tolist()
						params['l%d/b:0' % (i)] =                         selected['l%d/b_%d:0' % (i, task_id)]
	
	#                         params['l%d/w:0' % (i)] = np.concatenate([params['l%d/w:0' % (i)],params['l%d/w_%d:0' % (i,task_id)]], axis=1).tolist()
	#                         params['l%d/b:0' % (i)] = np.concatenate([params['l%d/b:0' % (i)],params['l%d/b_%d:0' % (i,task_id)]], axis=0).tolist()
					else:
						temp_weight = params['l%d/w:0' % i]
						temp_biases = params['l%d/b:0' % i]
						temp_weight[np.ix_(all_indices['l%d' % i], all_indices['l%d' % (i + 1)])] =                         selected['l%d/w:0' % i]
						temp_biases[all_indices['l%d' % (i + 1)]] =                         selected['l%d/b:0' % i]
						params['l%d/w:0' % i] = temp_weight
						params['l%d/b:0' % i] = temp_biases
	
				den.destroy_graph()
				den.restore_params(trainable=False, param_values=params)  # Freezes all learned weights making them non-trainable
	
				# ---------Performance-----------
				test_accuracy = den.predict(task_id=task_id, output_len=den.last_label_index+1)
				
				if test_accuracy < 0.85:
					
					print("Overall accuracy: %g, lesser than threshold" %test_accuracy)
					print("-----------Started Dynamic Expansion------------")
					y_conv = den.build_model(task_id=task_id, expansion=True, output_len=den.last_label_index+1)
					den.optimization()
					_ = den.train_task(task_id=task_id, batch_size=den.batch_size, epochs=den.epochs, expansion=True)
					params = den.get_params()
					
					# deleting useless neurons added during expansion
					for i in reversed(range(1, den.den_layers)):
						useless = []
						prev_layer_weight = params['l%d/w:0'%i]
						prev_layer_biases = params['l%d/b:0'%i]
						
						for j in range(prev_layer_weight.shape[1] - den.k_ex, prev_layer_weight.shape[1]):
							if np.count_nonzero(prev_layer_weight[:, j]) == 0:
								useless.append(j)
						print(len(useless))

						params['l%d/w:0'%i] = np.delete(prev_layer_weight, useless, axis = 1)
						params['l%d/b:0'%i] = np.delete(prev_layer_biases, useless)
						
						next_layer_weight = params['l%d/w:0'%(i+1)]
						params['l%d/w:0'%(i+1)] = np.delete(next_layer_weight, useless, axis = 0)
					
					den.destroy_graph()
					den.restore_params(trainable=False, param_values=params)  # Freezes all learned weights making them non-trainable
					
					test_accuracy = den.predict(task_id=task_id, output_len=den.last_label_index+1)
			
			stop = timeit.default_timer()
			
			print("Overall accuracy: %g \n" %test_accuracy)
			temp.append(test_accuracy)
			time.append(stop - start)

			l2_n = den.get_variable(name="w", scope="l2")
			s.append(l2_n.get_shape().as_list())
		
		s_all.append(s)

		den.destroy_graph()
		den.sess.close()
		del den
		all_accuracies.append(temp)
		all_time.append(time)
		
	print(all_accuracies)
	print(all_time)
	print(s_all)
	
	np.save(np_all_accuracies_path, np.array(all_accuracies))
	np.save(np_all_time_path, np.array(all_time))
	np.save(np_s_all_path, np.array(s_all))

# In[ ]:





# In[ ]:




