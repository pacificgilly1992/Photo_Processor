__name__ = 'Image and Video Organiser'
__description__ = 'Fixes the naming of all videos and images within a directory\
					or folder by renaming all images and videos using the\
					subdirectory and age of file as a basis'
__author__ = 'James Gilmore'
__date__ = '02/02/2019'
__version__ = '0.1'

import os
import sys
import glob

import numpy as np
import scipy as sp
import pandas as pd

class image_processor(object):
	
	def __init__(self, basedir, images=True, videos=True):
		"""
		Description
		-----------
		
		Finds all files within basedir and renames all files within all 
		subdirectories with the filename format,
		
		image_<sub_directory>_imagenum.<original_file_format>
		
		The imagenum is sequential based on only the images in each sub-
		directory and is invariant of image file type. The sequential
		behaviour is based on the date the picture was created.
		
		Parameters
		----------
		basedir : str,
			The directory location you want to organize all images into.
			The direcotry location can be the absolute or relative path.
		images : boolean, optional, default = True
			Specify whether to process images within the subdirectories.
		videos : boolean, optional, default = True
			Specify whether to process videos within the subdirectories.
		"""
		
		# Prerequisites
		self.image_formats = ['.jpg', '.JPG', '.tiff', '.bmp', '.png']
		self.video_formats = ['.mp4', '.mov', '.avi']
		
		# Error checking
		if not isinstance(basedir, str): 
			raise ValueError("'basedir' must be a string defining the \
								location of all images and videos to \
								process")
								
		if not isinstance(images, bool):
			raise ValueError("'images' must be either True or False")

		if not isinstance(videos, bool):
			raise ValueError("'videos' must be either True or False")

		# Find all images and videos within basedir
		self._findfiles(basedir=basedir, images=images, videos=videos)
		
	def _findfiles(self, basedir, images, videos):
		"""
		Finds all images and videos within basedir
		"""
		
		# Locate all images in selected directory (e.g. .jpg, .JPG, 
		# .png, .tiff, .bmp)
		image_files = glob.glob(basedir + '*')

		# Locate all videos in selected directory (e.g. .avi, .mp4,
		# .mov)
		video_files = glob.glob(basedir + '*')
	
		print("images_files", image_files)
		print("video_files", video_files)

	def rename(self):
		"""
		Method used to rename all files within basedir. Follows the 
		filename format,
		
		image_<sub_directory>_imagenum.<original_file_format>
		"""

		return

if __name__ == '__main__':

	"Hello Friend"
