# THIS SCRIPT HANDLE SETTING UP TENSORFLOW MODEL

import tensorflow as tf

def build_model(image_size, ground_image_size, num_channels, f1=9,n1=64,n2=32,f3=5):

    images = tf.placeholder(tf.float32, [None, image_size, image_size, num_channels], name='images')
    ground_images = tf.placeholder(tf.float32, [None, ground_image_size, ground_image_size, num_channels], name='ground_images')
    initializer = tf.contrib.layers.xavier_initializer()

    weights = {
      'W1': tf.Variable(initializer([f1, f1, num_channels, n1]), name='W1'),
      'W2': tf.Variable(initializer([num_channels, num_channels, n1, n2]), name='W2'),
      'W3': tf.Variable(initializer([f3, f3, n2, num_channels]), name='W3')
    }

    biases = {
      'B1': tf.Variable(tf.zeros([n1]), name='B1'),
      'B2': tf.Variable(tf.zeros([n2]), name='B2'),
      'B3': tf.Variable(tf.zeros([num_channels]), name='B3')
    }

    conv_layer1 = tf.nn.relu(tf.nn.conv2d(images, weights['W1'], strides=[1,1,1,1], padding='SAME') + biases['B1'])
    conv_layer2 = tf.nn.relu(tf.nn.conv2d(conv_layer1, weights['W2'], strides=[1,1,1,1], padding='SAME') + biases['B2'])
    conv_layer3 = tf.nn.conv2d(conv_layer2, weights['W3'], strides=[1,1,1,1], padding='SAME', name='cnn') + biases['B3']

    saver = tf.train.Saver(max_to_keep=5)

    return {'images':images, 'ground_images':ground_images, 'cnn':conv_layer3, 'saver':saver}
