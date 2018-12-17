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

from testing import test

'''	Note: define it with a main and run in command line makes it difficult to debug and somehow redundant to parse the arguments.
	Since we are running the demo in a pynotebook as requested by Andrew, we can simply make this file a module.
	A mockup notebook on how to use database is provided in HDF5_example.ipynb. Simply instantiate the database and
	import "run" into the notebook, passing database as the input arg.

'''
def run(database, option='train', learning_rate=1e-4, num_epoch=10, batch_size=1, saved_data_file_name=None):
	'''
		Lets'define the inputs here
	'''
	print("Preparing data...")
	database.initialize()

	if option == 'train':

		sess = tf.Session()

		model = build_model(database.patch_shape[0], database.patch_shape[0], database.patch_shape[2])

        # MSE loss function
		loss = tf.reduce_mean(tf.square(model['ground_images'] - model['cnn']))

		optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

		sess.run(tf.global_variables_initializer())

		step = 0

		for epoch in range(num_epoch):

			num_batches = database.size('train') // batch_size
			loss_sum = 0

			for idx in tqdm(range(num_batches)):
				step += 1
				batch_images, batch_ground_images = database['train',idx*batch_size:(idx+1)*batch_size]
				batch_images = batch_images.astype(np.float32)/255
				batch_ground_images = batch_ground_images.astype(np.float32)/255
				try:
					loss_sum += train(batch_images, batch_ground_images, model, optimizer, loss, step, sess)
				except:
					#print([idx*batch_size,(idx+1)*batch_size])
					#print(batch_images.shape)
					#print(batch_ground_images.shape)
					raise Exception('break')
			avg_train_mse = loss_sum/num_batches
			print("EPOCH:", epoch,"Avg Train Loss - :",10*np.log10(1/avg_train_mse)," db")

	else:

		sess = tf.Session()

		saver = tf.train.import_meta_graph('saved_model/' + saved_data_file_name)

		saver.restore(sess,tf.train.latest_checkpoint('saved_model'))

		graph = tf.get_default_graph()

		input_images = graph.get_tensor_by_name("images:0")

		cnn = graph.get_tensor_by_name("cnn:0")
