# THIS SCRIPT HANDLE MODEL TRAINING

from build_model import build_model

import tensorflow as tf

import shutil, sys

def train(images, ground_images, model, optimizer, loss, step, sess):

    # The "model" parameter is the output of the build_model() function in build_model.py

    _, loss_val = sess.run([optimizer,loss], feed_dict={model['images']:images,model['ground_images']:ground_images})

    #Save after every 100 steps:
    if step % 100 == 0:
        f = open('best_loss.txt', 'r')
        best_loss = float(f.read())
        f.close()
        if loss_val < best_loss:
            sys.stdout.flush()
            shutil.rmtree('saved_model')
            f = open('best_loss.txt', 'w')
            f.write('%f' %(loss_val))
            f.close()
            saver = model['saver']
            saver.save(sess, 'saved_model/model', global_step=step)


    return loss_val
