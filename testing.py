# THIS SCRIPT HANDLE MODEL TESTING

import tensorflow as tf

def test(images, ground_images, model, sess,loss):
	val_loss = sess.run(loss, feed_dict={model['images']:images,model['ground_images']:ground_images})
	return val_loss
    
