import os
import numpy as np
import tensorflow as tf
from scipy.misc.pilutil import imread, imresize
import matplotlib.pyplot as plt

def test_acc(sess, test_x, test_y):
    test_batch = int(len(test_x)/batch_size)
    acc = 0
    for i in range(test_batch):
        batch_xs = test_x[i * batch_size:(i + 1) * batch_size]
        batch_ys = test_y[i * batch_size:(i + 1) * batch_size]
        test_acc = sess.run(accuracy, feed_dict = {X:batch_xs, Y: batch_ys, keep_prob:1.0})
        acc += test_acc
    return acc/test_batch

def make_flip_img(x_data, y_data):
    raw_y = y_data[:]
    flip1 = np.flip(x_data,1)
    flip2 = np.flip(x_data,2)
    flip3 = np.flip(flip1, 2)
    x_data = np.concatenate((x_data,flip1), axis=0)
    x_data = np.concatenate((x_data,flip2), axis=0)
    x_data = np.concatenate((x_data,flip3), axis=0)
    y_data = np.concatenate((y_data,raw_y), axis=0)
    y_data = np.concatenate((y_data,raw_y), axis=0)
    y_data = np.concatenate((y_data,raw_y), axis=0)
    return x_data, y_data

def rgb2gray(img_data):
    for data in img_data:
        if len(data.shape) is 3:
            data = np.dot(data[...,:3], [0.299, 0.587, 0.114])
        else:
            data = data
    return img_data



data = np.load("dataset.npz")
train_img = data["train_img"]
train_label = data["train_label"]
test_img = data["test_img"]
test_label = data["test_label"]

total_img = np.concatenate([train_img, test_img],0)
total_label = np.concatenate([train_label, test_label],0)

total_img = np.array(total_img)/255
ont_hot = [[1,0] if i =="cat" else [0,1] for i in total_label]

np.random.seed(1400)
randidx = np.random.randint(len(total_img), size=len(total_img))

one_hot = np.array(ont_hot)
shuffle_x = total_img[randidx,:]
shuffle_y = one_hot[randidx,:]

split_value = int(0.8*len(shuffle_x))

train_x  = shuffle_x[:split_value]
test_x = shuffle_x[split_value:]

train_y  = shuffle_y[:split_value]
test_y = shuffle_y[split_value:]

train_x, train_y = make_flip_img(train_x, train_y)
train_x = rgb2gray(train_x)


training_epochs = 20
batch_size = 64
learning_rate = 0.001

X = tf.placeholder(tf.float32, shape=[None, 256, 256, 3])
Y = tf.placeholder(tf.float32, shape=[None, 2])

keep_prob = tf.placeholder(tf.float32)

#conv layer 1

W1 = tf.get_variable("w1", shape = [3,3,3,64], dtype = tf.float32, initializer=tf.contrib.layers.xavier_initializer_conv2d())
conv1 = tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding="SAME",use_cudnn_on_gpu=True)
b1 = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable = True, name='b1')
conv1 = tf.nn.relu(tf.nn.bias_add(conv1, b1))

#conv layer 2
W2 = tf.get_variable("w2", shape = [1, 1, 64, 64], dtype = tf.float32, initializer=tf.contrib.layers.xavier_initializer_conv2d())
conv2 = tf.nn.conv2d(conv1, W2, strides=[1,2,2,1], padding='SAME',use_cudnn_on_gpu=True)
b2 = tf.Variable(tf.constant(0.0, shape=[64], dtype = tf.float32), trainable=True, name='b2')
conv2 = tf.nn.relu(tf.nn.bias_add(conv2, b2))
max_pool2 = tf.nn.max_pool(conv2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

#conv layer 3
W3 = tf.get_variable("w3", shape = [3, 1 , 64, 64], dtype = tf.float32, initializer=tf.contrib.layers.xavier_initializer_conv2d())
conv3 = tf.nn.conv2d(max_pool2, W3, strides=[1,1,1,1], padding='SAME',use_cudnn_on_gpu=True)
b3 = tf.Variable(tf.constant(0.0, shape=[64], dtype = tf.float32), trainable=True, name='b3')
conv3 = tf.nn.relu(tf.nn.bias_add(conv3, b3))

W31 = tf.get_variable("w31", shape = [1, 3 , 64, 64], dtype = tf.float32, initializer=tf.contrib.layers.xavier_initializer_conv2d())
conv31 = tf.nn.conv2d(conv3, W31, strides=[1,1,1,1], padding='SAME',use_cudnn_on_gpu=True)
b31 = tf.Variable(tf.constant(0.0, shape=[64], dtype = tf.float32), trainable=True, name='b31')
conv31 = tf.nn.relu(tf.nn.bias_add(conv31, b31))

#conv  layer 4
W4 = tf.get_variable("w4", shape = [1, 1 , 64, 128], dtype = tf.float32, initializer=tf.contrib.layers.xavier_initializer_conv2d())
conv4 = tf.nn.conv2d(conv31, W4, strides=[1,1,1,1], padding='SAME',use_cudnn_on_gpu=True)
b4 = tf.Variable(tf.constant(0.0, shape=[128], dtype = tf.float32), trainable=True, name='b4')
conv4 = tf.nn.relu(tf.nn.bias_add(conv4, b4))
max_pool4 = tf.nn.max_pool(conv4, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')


#fully_connected1
flat = tf.reshape(max_pool4, [-1, 32*32*128])
W11 = tf.get_variable("w11", shape = [32*32*128, 512], dtype = tf.float32, initializer=tf.contrib.layers.xavier_initializer_conv2d())
b11 = tf.Variable(tf.constant(0.0, shape=[512], dtype = tf.float32), trainable=True, name='b11')
fc11 = tf.nn.relu(tf.nn.bias_add(tf.matmul(flat,W11), b11))
fc11 = tf.nn.dropout(fc11, keep_prob=keep_prob)

#fully_connected2
W12 = tf.get_variable("w12", shape = [512,128], dtype = tf.float32, initializer=tf.contrib.layers.xavier_initializer_conv2d())
b12 = tf.Variable(tf.constant(0.0, shape=[128], dtype = tf.float32), trainable=True, name='b12')
fc12 = tf.nn.relu(tf.nn.bias_add(tf.matmul(fc11,W12), b12))
fc12 = tf.nn.dropout(fc12, keep_prob=keep_prob)

#fully_connected3
W13 = tf.get_variable("w13", shape = [128,2], dtype = tf.float32, initializer=tf.contrib.layers.xavier_initializer_conv2d())
b13 = tf.Variable(tf.constant(0.0, shape=[2], dtype = tf.float32), trainable=True, name='b13')
logits = tf.matmul(fc12, W13) + b13


# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("Learning Started")
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(len(train_x) / batch_size)
        for i in range(total_batch):
            batch_xs = train_x[i * batch_size:(i + 1) * batch_size]
            batch_ys = train_y[i * batch_size:(i + 1) * batch_size]
            c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.5})
            avg_cost += c / total_batch

        print("Epoch :", epoch + 1, "cost :", avg_cost)
    test_acc = test_acc(sess, test_x, test_y)

    print("Accuracy :", test_acc)