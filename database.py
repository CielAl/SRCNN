# THIS SCRIPT HANDLES DATA PROCESSING
''' Requirements: H5 and Table
	keyname: img, label
	
'''
import tables
import glob

import os
import sys

#image io and processing
import cv2
import PIL
import numpy as np

#patch extraction and dataset split: Use 1 pair of train-validation to due to limitation of time
from sklearn import model_selection
import sklearn.feature_extraction.image
from sklearn.feature_extraction.image import extract_patches
import random

from tqdm import tqdm
from types import SimpleNamespace


class database(object):


	''' The constructor of the class. Some of the logic are inspired from the blog and its corresponding code here:
			http://www.andrewjanowczyk.com/pytorch-unet-for-digital-pathology-segmentation/
		Args:
			Mandatory:
				filedir: base dir of the image
				
				database_name: file name of the database
				export_dir: location to write the table
				patch_shape: Tuple. shape of the patch. Either (size,size,channel) or (size,size) in case of 1-channel images.
				stride_size: overlapping of patches.
			Optional:
				pattern: wildcard pattern of the images. Default is *.jpg
				interp: the interpolation method, default is PIL.IMAGE.NONE
				resize: the factor of resize the processing, which is 1/downsample_factor.
				dtype:  data type to be stored in the pytable. Default: UInt8Atom
				test_ratio: ratio of the dataset as test. Default: 0.1
		Raises:
			KeyError if the mandatory inputs is missing
	'''
	def __init__(self,**kwargs):
		self.filedir = kwargs['filedir']
		
		self.database_name = kwargs['database_name']
		self.export_dir = kwargs['export_dir']
		
		self.patch_shape = kwargs['patch_shape']
		self.stride_size = kwargs['stride_size']
		
		
		self.pattern = kwargs.get('pattern','*.jpg')
		self.interp = kwargs.get('interp',PIL.Image.NONE)
		self.resize = kwargs.get('resize',0.5)
		self.dtype =  kwargs.get('dtype',tables.UInt8Atom())
		self.test_ratio = kwargs.get('test_ratio',0.1)
		
		
		self.filenameAtom = tables.StringAtom(itemsize=255)

		self.filelist = self.get_filelist()
		#for now just take 1 set of train-val shuffle. Leave the n_splits here for future use.
		self.phases = self.init_split()
		self.types = ['img','label']

	def get_filelist(self):
		file_pattern = os.path.join(self.filedir,self.pattern)
		files=glob.glob(file_pattern)
		return files

	def init_split(self):
		phases = {}
		phases['train'],phases['val'] = next(iter(model_selection.ShuffleSplit(n_splits=10,test_size=self.test_ratio).split(self.filelist)))
		return phases

	def img_label_pair(self,file):
		img = cv2.imread(file,cv2.COLOR_BGR2RGB)
		img_down = cv2.resize(img,(0,0),fx=self.resize,fy=self.resize, interpolation=self.interp)
		#the shape is (y,x) while cv2.resize requires (x,y)
		img_down = cv2.resize(img_down,(img.shape[1],img.shape[0]),interpolation=self.interp)
		return img,img_down

	def generate_patch(self,image):
		patches_label= extract_patches(image,self.patch_shape,self.stride_size)
		patches_label = patches_label.reshape((-1,)+self.patch_shape)
		return patches_label
	
	def generate_tablename(self,phase):
		pytable_dir = os.path.join(self.export_dir)
		pytable_fullpath = os.path.join(pytable_dir,"%s_%s%s" %(self.database_name,phase,'.pytable'))
		return pytable_fullpath,pytable_dir
		
	# Tutorial from  https://github.com/jvanvugt/pytorch-unet
	def write_data(self):
		h5arrays = {}
		debug = {}
		filters=tables.Filters(complevel= 5)
		types = self.types
		#for each phase create a pytable
		self.tablename = {}
		pytable = {}
		for phase in self.phases.keys():
			#export dir  -- use normal formatted string so it can be run on python3.6
			#pytable_dir = os.path.join(self.export_dir)
			#pytable_fullpath = os.path.join(pytable_dir,"%s_%s%s" %(self.database_name,phase,'.pytable'))
			#self.tablename[phase] = pytable_fullpath
			pytable_fullpath,pytable_dir = self.generate_tablename(phase)
			if not os.path.exists(pytable_dir):
				os.makedirs(pytable_dir)
			pytable[phase] = tables.open_file(pytable_fullpath, mode='w')
			#debug[phase] = pytable
			h5arrays['filename'] = pytable[phase].create_earray(pytable[phase].root, 'filename', self.filenameAtom, (0,))

			for type in types:
				h5arrays[type]= pytable[phase].create_earray(pytable[phase].root, type, self.dtype,
													  shape=np.append([0],self.patch_shape),
													  chunkshape=np.append([1],self.patch_shape),
													  filters=filters)

			#
			#cv2.COLOR_BGR2RGB
			for file_id in tqdm(self.phases[phase]):
				#img as label,
				file = self.filelist[file_id]
				
				img_truth,img_down = self.img_label_pair(file)

				patches = {}
				patches[types[0]] = self.generate_patch(img_down)
				patches[types[1]] = self.generate_patch(img_truth)
				if patches[types[0]].shape[0]!= patches[types[1]].shape[0]:
					print(patches[types[0]].shape)
					print(patches[types[1]].shape)
					raise Exception(file)
				for type in types:
					h5arrays[type].append(patches[type])
					debug[type] = debug.get(type,0)+patches[type].shape[0]

			h5arrays["filename"].append([file for x in range(patches[types[0]].shape[0])])
			for k,v in pytable.items():
				v.close()
		return debug
	def is_instantiated(self,phase):
		file_path = self.generate_tablename(phase)[0]
		return os.path.exists(file_path)
	
	def initialize(self):
		if (not self.is_instantiated('train')) or (not self.is_instantiated('val')):
			self.write_data()
			
	'''
		Slice the chunk of patches out of the database.
		Precondition: Existence of pytable file. Does not require to call "write_data" as long as pytables exist
		Args:
			phase_index_tuple: ('train/val',index)
		Returns:
			images, labels(ground truth)
	'''
	def __getitem__(self, phase_index_tuple):
		phase,index = phase_index_tuple
		with tables.open_file(self.generate_tablename(phase)[0],'r') as pytable:
			image = pytable.root.img[index,]
			label = pytable.root.label[index,]
		return image,label

	def size(self,phase):
		with tables.open_file(self.generate_tablename(phase)[0],'r') as pytable:
			return pytable.root.img.shape[0]
	
	def peek(self,phase):
		with tables.open_file(self.generate_tablename(phase)[0],'r') as pytable:
			return pytable.root.img.shape,pytable.root.label.shape
