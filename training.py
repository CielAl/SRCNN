# THIS SCRIPT HANDLE MODEL TRAINING

from build_model import build_model

import tensorflow as tf

def train(images, ground_images, model, sess, learning_rate=1, num_epoch=1, batch_size=1):

    # The "model" parameter is the output of the build_model() function in build_model.py

    # MSE loss function
    loss = tf.reduce_mean(tf.square(model['ground_images'] - model['cnn']))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    tf.initialize_all_variables().run()

    saver = tf.train.Saver(max_to_keep=5)

    for epoch in range(num_epoch):
        print("EPOCH:", epoch)
        num_batches = len(images) // batch_size
        for batch in range(num_batches):
            batch_images = images[batch*batch_size : (batch+1)*batch_size]
            batch_ground_images = ground_images[batch*batch_size : (batch+1)*batch_size]

            sess.run(optimizer, feed_dict={model['images']:batch_images,model['ground_images']:batch_ground_images})
        # Save after every 5 epoches:
        if epoch % 5 == 0:
            saver.save(sess, 'saved_model/model.ckpt', global_step=epoch)
    return
