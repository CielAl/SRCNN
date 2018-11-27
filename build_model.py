# THIS SCRIPT HANDLE SETTING UP TENSORFLOW MODEL

import tensorflow as tf

def build_model(image_size, ground_image_size, num_channels, f1=9,n1=64,n2=32,f3=5):

    images = tf.placeholder(tf.int8, [None, image_size, image_size, num_channels], name='images')
    ground_images = tf.placeholder(tf.int8, [None, ground_image_size, ground_image_size, num_channels], name='ground_images')

    weights = {
      'W1': tf.Variable(tf.random_normal([f1, f1, 1, n1], stddev=1e-3), name='W1'),
      'W2': tf.Variable(tf.random_normal([1, 1, n1, n2], stddev=1e-3), name='W2'),
      'W3': tf.Variable(tf.random_normal([f3, f3, n2, 1], stddev=1e-3), name='W3')
    }

    biases = {
      'B1': tf.Variable(tf.zeros([n1]), name='B1'),
      'B2': tf.Variable(tf.zeros([n2]), name='B2'),
      'B3': tf.Variable(tf.zeros([1]), name='B3')
    }

    conv_layer1 = tf.nn.relu(tf.nn.conv2d(images, weights['W1'], strides=[1,1,1,1], padding='VALID') + biases['B1'])
    conv_layer2 = tf.nn.relu(tf.nn.conv2d(conv1, weights['W2'], strides=[1,1,1,1], padding='VALID') + biases['B2'])
    conv_layer3 = tf.nn.conv2d(conv2, weights['W3'], strides=[1,1,1,1], padding='VALID') + self.biases['B3']

    return {'images':images, 'ground_images':ground_images, 'cnn':conv_layer3}
