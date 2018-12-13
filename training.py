# THIS SCRIPT HANDLE MODEL TRAINING

from build_model import build_model

import tensorflow as tf

def train(images, ground_images, model, optimizer, loss, epoch, sess):

    # The "model" parameter is the output of the build_model() function in build_model.py

    _, loss_val = sess.run([optimizer,loss], feed_dict={model['images']:images,model['ground_images']:ground_images})

    #Save after every 5 epoches:
    if (epoch+1) % 5 == 0:
        saver = model['saver']
        saver.save(sess, 'saved_model/model.ckpt', global_step=epoch)
        f = open('saved_model/log.txt','a')
        f.write('Epoch: %f.Loss = %f\n' %(epoch, loss_val))
