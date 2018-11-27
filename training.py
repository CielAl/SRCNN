# THIS SCRIPT HANDLE MODEL TRAINING

from build_model import build_model

import tensorflow as tf

def train(images, ground_images, model, learning_rate, num_epoch, batch_size, sess):

    # The "model" parameter is the output of the build_model() function in build_model.py

    # Code block for getting data: assume 4-D tensor, tf.uint8, images of the same size, ground_images of the same size
    images, ground_images = get_data() # Will implement after Yufei finishes data processing

    # MSE loss function
    loss = tf.reduce_mean(tf.square(model['ground_images'] - model['cnn']))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    tf.initialize_all_variables().run()

    for epoch in range(num_epoch):
        num_batches = len(images) // batch_size
        for batch in range(num_batches):
            batch_images = images[batch*batch_size : (batch+1)*batch_size]
            batch_ground_images = ground_images[batch*batch_size : (batch+1)*batch_size]

            sess.run(optimizer, feed_dict={model['images']:batch_images,model['ground_images']:batch_ground_images})

    return model['cnn']
