#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from collections import defaultdict
import gzip
import cv2 as cv
import os
import numpy as np


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
        self.selected = dict()
        self.l2_mu = 0.01
        self.lamba_regular = 0.5
        self.l1_thr = 0.00001
        self.loss_thr = 0.01

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

            if task_id != None:
                if 'l%d/w_%d' % (self.den_layers,task_id) in scope_name:
                    trainable = True
                else:
                    trainable = False

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

            self.h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

            # readout fc layer
            self.w_fc = self.create_variable(name="w", shape=[128, output_len], scope="l3")     # layer-3 outgoing weight matrix 
            self.b_fc = self.create_variable(name="b", shape=[output_len], scope="l3")
            y_conv = tf.matmul(self.h_fc2, self.w_fc) + self.b_fc
            
        elif expansion:
            
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
                    h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, w_expanded) + b_expanded)

                elif layer == self.den_layers:
                    w_fc3 = self.get_variable(name="w_%d"%task_id,scope="l%d"%layer)
                    b_fc3 = self.get_variable(name="b_%d"%task_id,scope="l%d"%layer)
                    
                    prev_dim = w_fc3.get_shape().as_list()[0]
                    next_dim = w_fc3.get_shape().as_list()[1]

                    w_expand = self.create_variable(name="w_ex_"+str(task_id), shape=[prev_dim + self.k_ex, next_dim], scope="l%d"%layer)
                    
                    w_expanded = tf.concat([w_fc3, w_expand], 0)
                    
                    self.params[w_fc3.name] = w_expanded
                    y_conv = tf.matmul(self.h_fc2, w_expanded) + b_fc3

                else:
                    w_fc2 = self.get_variable(name="w",scope="l%d"%layer)
                    b_fc2 = self.get_variable(name="b",scope="l%d"%layer)
                    prev_dim = w_fc2.get_shape().as_list()[0]
                    next_dim = w_fc2.get_shape().as_list()[1]

                    dummy_w = tf.get_variable(name="w_ex_d_"+str(task_id), shape=[self.k_ex, next_dim], 
                                initializer=tf.constant_initializer(0.0), scope="l%d"%layer, trainable=False)

                    w_expand = self.create_variable(name="w_ex_"+str(task_id), shape=[prev_dim + self.k_ex, self.k_ex], scope="l%d"%layer)
                    b_expand = self.create_variable(name="b_ex_"+str(task_id), shape=[self.k_ex], scope="l%d"%layer)
                    
                    w_fc2 = tf.concat([w_fc2, dummy_w],0)
                    
                    w_expanded = tf.concat([w_fc2, w_expand], 1)
                    b_expanded = tf.concat([b_fc2, b_expand], 0)
                    
                    self.params[w_fc2.name] = w_expanded
                    self.params[b_fc2.name] = b_expanded
                    self.h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_expanded) + b_expanded)
                    
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

            self.h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

            self.w_fc = self.create_variable(name="w_"+str(task_id), shape=[128, output_len], scope="l3", trainable=True)     # layer-3 outgoing weight matrix 
            self.b_fc = self.create_variable(name="b_"+str(task_id), shape=[output_len], scope="l3", trainable=True)
            y_conv = tf.matmul(self.h_fc2, self.w_fc) + self.b_fc

#         y_conv = tf.nn.sigmoid(y_conv)
        return y_conv
    
    def perform_selection(self, task_id, values_dict):      # Breadth first search for selecting non-zero units
        
        all_indices = defaultdict(list)     # to store indices of nonzero units
        selected_params = dict()            # to store values of nonzero units
        selected_prev_params = dict()
        
        for scope, value in values_dict.items():    # Storing conv layers in selected parameters 
            if "conv" in scope:
                selected_params[scope] = value
            
        for i in reversed(range(1,self.den_layers+1)):
            if i == self.den_layers:
                w = values_dict['l%d/w_%d:0' %(i,task_id)]
                b = values_dict['l%d/b_%d:0' %(i,task_id)]
                for j in range(w.shape[0]):
                    if w[j,0] != 0:
                        all_indices['l%d' % i].append(j)
                # np.ix_(): fancy indexing, index with arrays of integers
                # Select non-zero weights between the last hidden layer and the output layer
                selected_params['l%d/w_%d:0' % (i, task_id)] =                     w[np.ix_(all_indices['l%d' % i], [0])]
                selected_params['l%d/b_%d:0' % (i, task_id)] = b
            else:
                w = values_dict['l%d/w:0' % i]
                b = values_dict['l%d/b:0' % i]
                top_indices = all_indices['l%d' % (i + 1)]
                print(len(top_indices))
                for j in range(w.shape[0]):
                    if np.count_nonzero(w[j, top_indices]) != 0 or i == 1:
                        all_indices['l%d' % i].append(j)
                
                # non-zero weights between the layer i and the layer i+1
                sub_weight = w[np.ix_(all_indices['l%d' % i], top_indices)]
                sub_biases = b[all_indices['l%d' % (i + 1)]]
                selected_params['l%d/w:0' % i] = sub_weight
                selected_params['l%d/b:0' % i] = sub_biases
                
                # prev_W : to avoid drastic change in value of weights (Regularization)
                selected_prev_params['l%d/w:0' % i] =                     self.prev_W['l%d/w:0' % i][np.ix_(all_indices['l%d' % i], top_indices)]
                selected_prev_params['l%d/b:0' % i] =                     self.prev_W['l%d/b:0' % i][all_indices['l%d' % (i + 1)]]

#         for keys, value in selected_params.items():
#             print(keys)
#             print(value)
                
        return [selected_params, selected_prev_params, all_indices]
        
    def build_SR(self, task_id, selected, output_len):    # creating selective retraining model
        
        self.initialize_parameters(output_len)
        h = self.x
        
        # conv layers
        for i in range(1, self.conv_layers+1):
            with tf.variable_scope('conv%d' % i):
                w = tf.get_variable('w', initializer=selected['conv%d/w:0' % i], trainable=False)
                b = tf.get_variable('b', initializer=selected['conv%d/b:0' % i], trainable=False)
            h_conv = tf.nn.relu(self.conv2d(h, w) + b)
            h = self.max_pool_2x2(h_conv)
            
        # flattening
        h = tf.reshape(h, [-1, 15*15*64])
        
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
        tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_, logits=y_conv))

        for var in tf.trainable_variables():
            l2_regular = l2_regular + tf.nn.l2_loss(var)
            train_var.append(var)
#         print(len(train_var))

        self.loss = self.loss + tf.reduce_mean(self.l2_mu * l2_regular)

        if prev_W != None:
            for var in train_var:
                if var.name in prev_W.keys():
                    prev_w = prev_W[var.name]
                    regular_terms.append(tf.nn.l2_loss(var - prev_w))
            self.loss = self.loss + self.lamba_regular * tf.reduce_mean(regular_terms)
        
        opt = tf.train.AdamOptimizer(1e-5)
        grads = opt.compute_gradients(self.loss, train_var)
        apply_grads = opt.apply_gradients(grads)

        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(self.y_,1))
        self.acc_train = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        l1_var = [var for var in tf.trainable_variables()]
        l1_op_list = []
        with tf.control_dependencies([apply_grads]):  # exec apply_grads first
            for var in l1_var:
                th_t = tf.fill(tf.shape(var), tf.convert_to_tensor(self.l1_thr))
                zero_t = tf.zeros(tf.shape(var))
                var_temp = var - (th_t * tf.sign(var))
                # [pseudo]:  if |var| < th_t: var = [0];  else: var = var_temp
                l1_op = var.assign(tf.where(tf.less(tf.abs(var), th_t), zero_t, var_temp))
                l1_op_list.append(l1_op)
        
        with tf.control_dependencies(l1_op_list):
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

                if count > 2:
                    print("Best accuracy achieved! \n")
                    break
        
        for scope, ref in self.params.items():
            if "l2" in scope:
                print(self.sess.run(ref))
        
        return loss
    
    def predict(self, task_id, output_len=2):

        self.initialize_parameters(output_len)

        task_test_labels = tf.one_hot(indices=self.test_labels, depth=self.last_label_index+1)
        task_test_labels = self.sess.run(task_test_labels)
        
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

        self.h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

        self.w_fc = self.get_variable(name="w", scope="l3")
        self.b_fc = self.get_variable(name="b", scope="l3")

        y_final = tf.matmul(self.h_fc2, self.w_fc) + self.b_fc
        y_final = tf.nn.sigmoid(y_final)
        
        self.sess.run(tf.global_variables_initializer())

        correct_prediction = tf.equal(tf.argmax(y_final,1), tf.argmax(self.y_,1))
        self.acc_test = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        test_accuracy = self.acc_test.eval(session=self.sess, feed_dict={self.x: self.test, self.y_: task_test_labels})
        print("Overall accuracy: %g \n"%test_accuracy)
#         print("\n y_final:")
#         print(self.sess.run(y_final, feed_dict={self.x: self.test}))


# In[3]:


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
    epochs = 5
    early_stop = 5

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
        selected = dict()
        print("-------------Training new task: %d--------------"%task_id)
        if task_id == 1:
            y_conv = den.build_model(task_id=task_id)
            den.optimization()
            _ = den.train_task(task_id=task_id, batch_size=batch_size, epochs=epochs)
            params = den.get_params()
        else:
            print("-----------Started Selective Retraining-------------")
            
            #------------------Selection-------------------
            y_conv = den.build_model(task_id=task_id, output_len=1)
            den.optimization()
            
            print("---- Selecting nodes ----")
            _ = den.train_task(task_id=task_id, batch_size=batch_size, epochs=early_stop)
            params = den.get_params()
            [selected, selected_prev, all_indices] = den.perform_selection(task_id=task_id, values_dict=params)
            den.destroy_graph()
            
            #------------------Retraining-------------------
            print("---- Retraining selected nodes ----")
            y_conv = den.build_SR(task_id=task_id, selected=selected, output_len=1) 
            den.optimization(prev_W=selected_prev) 
            loss = den.train_task(task_id=task_id, batch_size=batch_size, epochs=epochs)
            
            print("Loss: %f"%loss)
            
#             if loss < den.loss_thr:
            if True:
            
                #--------------Performing Union----------------
                _vars = [(var.name, den.sess.run(var)) for var in tf.trainable_variables() if 'l' in var.name]

                for item in _vars:
                    key, values = item
                    selected[key] = values

                for i in reversed(range(1, den.den_layers+1)):
                    if i == den.den_layers:
                        temp_weight = params['l%d/w_%d:0' % (i, task_id)]
                        temp_weight[np.ix_(all_indices['l%d' % i], [0])] =                             selected['l%d/w_%d:0' % (i, task_id)]
                        params['l%d/w_%d:0' % (i, task_id)] = temp_weight
                        params['l%d/b_%d:0' % (i, task_id)] =                             selected['l%d/b_%d:0' % (i, task_id)]
                        # Updating output matrix structure
                        params['l%d/w:0' % (i)] = np.concatenate([params['l%d/w:0' % (i)],params['l%d/w_%d:0' % (i,task_id)]], axis=1).tolist()
                        params['l%d/b:0' % (i)] = np.concatenate([params['l%d/b:0' % (i)],params['l%d/b_%d:0' % (i,task_id)]], axis=0).tolist()
                    else:
                        temp_weight = params['l%d/w:0' % i]
                        temp_biases = params['l%d/b:0' % i]
                        temp_weight[np.ix_(all_indices['l%d' % i], all_indices['l%d' % (i + 1)])] =                             selected['l%d/w:0' % i]
                        temp_biases[all_indices['l%d' % (i + 1)]] =                             selected['l%d/b:0' % i]
                        params['l%d/w:0' % i] = temp_weight
                        params['l%d/b:0' % i] = temp_biases

#             else:
                
#                 print("\n-----------Started Dynamic Expansion------------")
#                 den.destroy_graph()
#                 den.restore_params(task_id=task_id, trainable=False, param_values=params)
#                 y_conv = den.build_model(task_id=task_id, expansion=True, output_len=1)
#                 den.optimization()
#                 _ = den.train_task(task_id=task_id, batch_size=batch_size, epochs=early_stop)
#                 params = den.get_params()

        den.destroy_graph()
        den.restore_params(trainable=False, param_values=params)    # Freezes all learned weights
                    
        # ---------Performance-----------
        den.predict(task_id=task_id, output_len=den.last_label_index+1)
                    
#         param_values = den.get_params()         # Backing-up model weights that were trained upto current task
       
    


# In[ ]:




