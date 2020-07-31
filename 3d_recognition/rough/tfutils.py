#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf


# In[2]:


seed = 1337
np.random.seed(seed)
tf.set_random_seed(seed)

batch_size = 64

w_init = tf.contrib.layers.variance_scaling_initializer(factor=1., mode='FAN_AVG', uniform=True)
b_init = tf.zeros_initializer()

reg = 5e-4
w_reg = tf.contrib.layers.l2_regularizer(reg)

eps = 1e-5

def conv2d(x, f=64, k=3, s=1, pad='SAME', reuse=None, is_train=True, name='conv2d'):
    """
    :param x: input
    :param f: filters
    :param k: kernel size
    :param s: strides
    :param pad: padding
    :param reuse: reusable
    :param is_train: trainable
    :param name: scope name
    :return: net
    """
    return tf.layers.conv2d(inputs=x,
                            filters=f, kernel_size=k, strides=s,
                            kernel_initializer=w_init,
                            kernel_regularizer=w_reg,
                            bias_initializer=b_init,
                            padding=pad,
                            reuse=reuse,
                            name=name)


def deconv2d(x, f=64, k=3, s=1, pad='SAME', reuse=None, name='deconv2d'):
    """
    :param x: input
    :param f: filters
    :param k: kernel size
    :param s: strides
    :param pad: padding
    :param reuse: reusable
    :param is_train: trainable
    :param name: scope name
    :return: net
    """
    return tf.layers.conv2d_transpose(inputs=x,
                                      filters=f, kernel_size=k, strides=s,
                                      kernel_initializer=w_init,
                                      kernel_regularizer=w_reg,
                                      bias_initializer=b_init,
                                      padding=pad,
                                      reuse=reuse,
                                      name=name)


def dense(x, f=1024, reuse=None, name='fc'):
    """
    :param x: input
    :param f: fully connected units
    :param reuse: reusable
    :param name: scope name
    :param is_train: trainable
    :return: net
    """
    return tf.layers.dense(inputs=x,
                           units=f,
                           kernel_initializer=w_init,
                           kernel_regularizer=w_reg,
                           bias_initializer=b_init,
                           reuse=reuse,
                           name=name)


def batch_norm(x, momentum=0.9, center=True, scaling=True, is_train=True, reuse=None, name="bn"):
    return tf.layers.batch_normalization(inputs=x,
                                         momentum=momentum,
                                         epsilon=eps,
                                         center=center,
                                         scale=scaling,
                                         training=is_train,
                                         reuse=reuse,
                                         name=name)

def sce_loss(data, label):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=data, labels=label))


# In[ ]:




