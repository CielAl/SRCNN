# THIS IS THE MAIN SCRIPT

import tensorflow as tf
import os
from training import train
from build_model import build_model


from database import database
from matplotlib import pyplot as plt
import sklearn.feature_extraction.image
from sklearn.feature_extraction.image import extract_patches
import tables

'''	Note: define it with a main and run in command line makes it difficult to debug and somehow redundant to parse the arguments.
	Since we are running the demo in a pynotebook as requested by Andrew, we can simply make this file a module.
	A mockup notebook on how to use database is provided in HDF5_example.ipynb. Simply instantiate the database and 
	import "run" into the notebook, passing database as the input arg.

'''
def run(database):
    if len(sys.argv) != 11:
        raise Exception("Inappropriate number of arguments. Require 9.")
    training = int(sys.argv[1])
    image_size = int(sys.argv[2])
    ground_image_size = int(sys.argv[3])
    num_channels = int(sys.argv[4])
    f1 = int(sys.argv[5])
    n1 = int(sys.argv[6])
    n2 = int(sys.argv[7])
    f3 = int(sys.argv[8])
    num_epoch = int(sys.argv[9])
    batch_size = int(sys.argv[10])

	''' 
		Lets'define the inputs here
	'''
	
    if training:

        saver = tf.train.Saver(max_to_keep=5)

        sess = tf.Session()

        model = build_model(image_size, ground_image_size, num_channels, f1=f1,n1=n1,n2=n2,f3=f3)
        # MSE loss function
        loss = tf.reduce_mean(tf.square(model['ground_images'] - model['cnn']))

        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

        tf.initialize_all_variables().run()
        
        for epoch in range(num_epoch):
            print("EPOCH:", epoch)
            num_batches = len(images) // batch_size
            for idx in range(num_batches):

        # Code block for getting data: assume 4-D tensor, tf.uint8, images of the same size, ground_images of the same size
                chunk_id = [idx*batch_size : (idx+1)*batch_size]
                batch_images, batch_ground_images = database['train',chunk_id] # Will implement after Yufei finishes data processing
                train(batch_images, batch_ground_images, model, optimizer, epoch, saver, sess)
    else:
        pass
