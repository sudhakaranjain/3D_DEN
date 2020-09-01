#!/usr/bin/env python
# coding: utf-8

# In[36]:


import tensorflow as tf
import gzip
import cv2 as cv
import os
import numpy as np


# In[37]:


def load_model():
    
    global mobilenet
    mobilenet = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, weights="imagenet")
    mobilenet.trainable = False
    mobilenet = tf.keras.Model(inputs=mobilenet.input, outputs=[mobilenet.layers[-2].output])
    
    saver = tf.train.import_meta_graph('../Saved_Models/Final_Model/3D_DEN.meta')
    saver.restore(sess,tf.train.latest_checkpoint('../Saved_Models/Final_Model'))
    graph = tf.get_default_graph()
    global x
    x = tf.placeholder(tf.float32, [None, 1280])
#     x = graph.get_tensor_by_name("Placeholder:0")
#     print(x)

    # #flattened first fc layer
    W_fc1 = graph.get_tensor_by_name("l1/w:0")		# layer-1 outgoing weight matrix
    b_fc1 = graph.get_tensor_by_name("l1/b:0")

    h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

    # #second fc layer
    W_fc2 = graph.get_tensor_by_name("l2/w:0")		# layer-2 outgoing weight matrix
    b_fc2 = graph.get_tensor_by_name("l2/b:0")

    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    w_fc = graph.get_tensor_by_name("l3/w:0")
    b_fc = graph.get_tensor_by_name("l3/b:0")

    global y_final
    y_final = tf.matmul(h_fc2, w_fc) + b_fc
    y_final = tf.nn.softmax(y_final)

    sess.run(tf.global_variables_initializer())


# In[38]:


def predict():
    
    test = []
    test_labels = []
    data_path = '../ModelNet40/temp'
    labels = np.load('../Saved_Models/Final_Model/final_labels.npy')
    
    for label in sorted(os.listdir(data_path)):
        label_path = os.path.join(data_path, label)

        for file in (os.listdir(label_path)):
            file_path = os.path.join(label_path, file)

            if str(file) == "test":
                for instance in sorted(os.listdir(file_path)):
                    instance_path = os.path.join(file_path, instance)
                    test_views = []

                    for img in sorted(os.listdir(instance_path)):
                        img_path = os.path.join(instance_path, img)
                        image = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
                        re_image = cv.resize(image, (IMG_SIZE, IMG_SIZE))
                        test_views.append(re_image)
                    image2 = cv.merge(test_views)
                    test.append(image2)
                    test_labels.append(label)
                    break
        break
    #                 cv.imshow("image", image)
    #                 cv.waitKey(0)
    #                 cv.destroyAllWindows()
    #                 cv.imwrite("./2.jpg", image1)
    #                 break
    
    test = np.array(test, dtype="float") / 255.0
    test_features = mobilenet.predict(test, batch_size=5)
    
    prediction = tf.argmax(y_final,1)
    p = prediction.eval(session=sess, feed_dict={x: test_features})
    
    print("Predicted: ", labels[p[0]])
    
    print("Actual: ", test_labels[0])
    
    # self.acc_test = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # test_accuracy = self.acc_test.eval(session=self.sess, feed_dict={self.x: self.test, self.y_: task_test_labels})
    # return test_accuracy


# In[40]:


IMG_SIZE = 128
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
sess = tf.Session()
load_model()
predict()


# In[ ]:




