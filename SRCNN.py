# THIS IS THE MAIN SCRIPT

import tensorflow as tf
import os
from training import train
from build_model import build_model


from database import database
from matplotlib import pyplot as plt
import sklearn.feature_extraction.image
from sklearn.feature_extraction.image import extract_patches
from tqdm import tqdm
import tables
import numpy as np

'''	Note: define it with a main and run in command line makes it difficult to debug and somehow redundant to parse the arguments.
	Since we are running the demo in a pynotebook as requested by Andrew, we can simply make this file a module.
	A mockup notebook on how to use database is provided in HDF5_example.ipynb. Simply instantiate the database and
	import "run" into the notebook, passing database as the input arg.

'''
def run(database, option='train', learning_rate=1e-4, num_epoch=10, batch_size=1):
    # if len(sys.argv) != 11:
    #     raise Exception("Inappropriate number of arguments. Require 9.")
    # training = int(sys.argv[1])
    # image_size = int(sys.argv[2])
    # ground_image_size = int(sys.argv[3])
    # num_channels = int(sys.argv[4])
    # f1 = int(sys.argv[5])
    # n1 = int(sys.argv[6])
    # n2 = int(sys.argv[7])
    # f3 = int(sys.argv[8])
    # num_epoch = int(sys.argv[9])
    # batch_size = int(sys.argv[10])

	'''
		Lets'define the inputs here
	'''
	print("Preparing data...")
	#database.write_data()
	database.initialize()
	
	if option == 'train':

		sess = tf.Session()

		model = build_model(database.patch_shape[0], database.patch_shape[0], database.patch_shape[2])
        # MSE loss function
		loss = tf.reduce_mean(tf.square(model['ground_images'] - model['cnn']))

		optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

		tf.initialize_all_variables().run(session=sess)

		step = 0

		for epoch in range(num_epoch):
			

			num_batches = database.size('train') // batch_size
			loss_sum = 0
			for idx in tqdm(range(num_batches)):
        # Code block for getting data: assume 4-D tensor, tf.uint8, images of the same size, ground_images of the same size
				#chunk_id = (idx*batch_size,(idx+1)*batch_size)
				step += 1
				batch_images, batch_ground_images = database['train',idx*batch_size:(idx+1)*batch_size] # Will implement after Yufei finishes data processing
				#batch_images = tf.convert_to_tensor(batch_images,dtype=tf.float32)
				#batch_ground_images = tf.convert_to_tensor(batch_ground_images,dtype=tf.float32)
				batch_images = batch_images.astype(np.float32)/255
				batch_ground_images = batch_ground_images.astype(np.float32)/255
				try:
					loss_sum += train(batch_images, batch_ground_images, model, optimizer, loss, step, sess)
				except:
					print([idx*batch_size,(idx+1)*batch_size])
					print(batch_images.shape)
					print(batch_ground_images.shape)	
					raise Exception('break')
			
			print("EPOCH:", epoch,"Avg Loss:",loss_sum/num_batches)

	else:
		pass
