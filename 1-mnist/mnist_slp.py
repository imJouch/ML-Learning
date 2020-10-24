# -*- coding: utf-8 -*-
import tensorflow as tf
from input_data import read_data_sets
import os

# don't show INFO 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# read mnist
mnist = read_data_sets('MNIST_data', one_hot=True)


# single layer perceptron: y = wx + b
# input
x = tf.placeholder(tf.float32, [None, 784])

# weights
W = tf.Variable(tf.random_normal([784,10], stddev=0.1))

# bias
b = tf.Variable(tf.zeros([10]))

# softmax 
y = tf.nn.softmax(tf.matmul(x,W) + b)

# output
y_ = tf.placeholder(tf.float32, [None, 10])

# cross_entropy loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# optimization with gradient descend, the learning rate is set as 0.02
train_step = tf.train.GradientDescentOptimizer(0.02).minimize(cross_entropy)

# initalize all variables
init = tf.global_variables_initializer()

# start a new session
sess = tf.Session()
sess.run(init)

m_saver = tf.train.Saver()

# 2000 iterations
for i in range(2000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if i % 100 == 0:
        m_saver.save(sess, './model/mnist_slp', global_step=i)

# computer the accuracy
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

# close the session
sess.close()
