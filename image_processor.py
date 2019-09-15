__title__ = 'Image and Video Organiser'
__description__ = 'Fixes the naming of all videos and images within a directory\
                    or folder by renaming all images and videos using the\
                    subdirectory and age of file as a basis'
__author__ = 'James Gilmore'
__date__ = '13/09/2019'
__version__ = '0.5'

import glob
import os
import random
import string
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from gilly_utilities import progress
from PIL import Image

"""
New features to work on:
* Addin hashing scheme to check for duplicate images.
* Automatically choose the photo/video ID string based on the number of
    photos/videos that need processing (i.e. len(photos) > 1000 but less 
    than 10,000, then 4 digits are required to keep the filenames of 
    equal length.
* Add support for how to rename the photo/videos. At the moment we use 
    a common fileformt ==> 
        image_<sub_directory>_<imagenum>.<original_file_format>)
* Need to add support for nested directories. Current only works when
    all assets are in the same directory (I think).
"""

class image_processor(object):

    def __init__(self, basedir, images=True, videos=True):
        """
        Finds all files within basedir and renames all files within all 
        subdirectories with the filename format,
        
        image_<sub_directory>_<imagenum>.<original_file_format>
        
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
        self.image_formats = ['.jpg', '.JPG', '.tiff', '.bmp', '.png', '.PNG', 
            '.gif', '.GIF', '.jpeg']
        self.video_formats = ['.mp4', '.mov', '.MOV', '.avi']

        # Error checking
        if not isinstance(basedir, str): 
            raise ValueError("'basedir' must be a string defining the \
                                location of all images and videos to \
                                process")

        if not isinstance(images, bool):
            raise ValueError("'images' must be either True or False")

        if not isinstance(videos, bool):
            raise ValueError("'videos' must be either True or False")

        if not os.path.isdir(basedir):
            raise IOError("'basedir' must link to an existing directory")

        # Ensure correct format of basedir
        basedir = os.path.abspath(basedir)

        # Find all images and videos within basedir
        self.image_files, self.video_files = self._findfiles(basedir=basedir, 
                                            images=images, videos=videos)

        print("[INFO] Processing {photo} photos and {video} videos".format(
            photo=len(self.image_files), video=len(self.video_files)))

    def _findfiles(self, basedir, images, videos):
        """
        Finds all images and videos within basedir
        """

        # Initalise the Path class
        path = Path(basedir)

        # Locate all files in selected directory
        files = list(path.glob('**/*'))

        if images is True:

            # Subset image files for specific extension (e.g. .jpg, .JPG, 
            # .png, .tiff, .bmp)
            image_files = self._subset_files(files, self.image_formats)

        else:

            image_files = None

        if videos is True:

            # Subset video files for specific extension (e.g. .avi, .mp4,
            # .mov)
            video_files = self._subset_files(files, self.video_formats)

        else:

            video_files = None

        return image_files, video_files

    def _subset_files(self, filelist, extensions):
        """
        Subset filelist with extensions specified
        """

        # Search in each file if the extension matches the end of the file
        files = []
        for file in filelist:
            if file.suffix in extensions:
                files.append(file)

        # Return a subsetted filelist
        return files

    def _get_creation(self, file_list, file_type):
        """
        Finds the creation time for image and video files

        Parameters
        ----------
        filetypes : str
            specify the file types in filelist. Either 'images' or 'videos'

        Reference
        ---------
        https://stackoverflow.com/a/23064792
        """

        if file_type == 'images':
            creation_date = []
            with progress('Photos checked: %s/%s', size=len(file_list)) as prog:
                for i, file in enumerate(file_list):
                    
                    # Update the progress bar.
                    prog.update(i, len(file_list))

                    try:
                        creation_date.append(Image.open(file)._getexif()[36867])
                    except (KeyError, TypeError, AttributeError):
                        # Only works on Windows
                        creation_date.append(
                            datetime.fromtimestamp(
                                os.path.getmtime(file)).strftime('%Y:%m:%d %H:%M:%S'))

        elif file_type == 'videos':
            creation_date = []
            with progress('Videos checked: %s/%s', size=len(file_list)) as prog:
                for i, file in enumerate(file_list):

                    # Update the progress bar.
                    prog.update(i, len(file_list))

                    creation_date.append(
                                datetime.fromtimestamp(
                                    os.path.getmtime(file)).strftime('%Y:%m:%d %H:%M:%S'))

        return creation_date

    def _generate_temp_name(self, N=8):
        """
        Generate a tempory name by creating a random set of characters.

        :param N : int, optional
            Specify the number of characters you want the temporary name
            to be.
        """

        return ''.join(random.SystemRandom().choice(
            string.ascii_uppercase + string.digits) for _ in range(N))

    def rename(self):
        """
        Method used to rename all files within basedir. Follows the 
        filename format,
        
        image_<sub_directory>_imagenum.<original_file_format>
        """

        print("[INFO] Renaming your photos and videos.")

        # Convert self.image_files and self.video_files to numpy
        if self.image_files is not None:
            self.image_files = np.array(self.image_files, dtype=object)

        if self.video_files is not None:
            self.video_files = np.array(self.video_files, dtype=object)

        print("[INFO] Getting the creation dates of all your photos...")

        # Get creation dates for all image files
        image_creation_date = self._get_creation(self.image_files, 
            file_type='images')

        # Sort image_files by date
        mask = np.argsort(image_creation_date, kind='mergesort')
        self.image_files = self.image_files[mask]

        print("[INFO] Getting the creation dates of all your videos...")

        # Get creation dates for all video files
        video_creation_date = self._get_creation(self.video_files, 
            file_type='videos')

        # Sort image_files by date
        mask = np.argsort(video_creation_date, kind='mergesort')
        self.video_files = self.video_files[mask]

        print("[INFO] Creating tempory names for you photos and videos. " \
            "This avoids errors if you run this program multiple times.")

        # Rename all images/videos (FAKE Version)
        total_len = len(self.image_files) + len(self.video_files)
        iter = 0
        with progress('Videos checked: %s/%s', size=total_len) as prog:
            for file_type, file_supname in zip([self.image_files, self.video_files],
                    ['Image', 'Video']):
                temp_names = []
                
                for i, file in enumerate(file_type):
                    
                    # Update the progress bar.
                    prog.update(iter, total_len)

                    # Create new filename
                    temp_names.append(file.parents[0] 
                        / (self._generate_temp_name() + file.suffix))

                    # Rename file
                    os.rename(file, temp_names[-1])

                    iter += 1

                if file_supname == 'Image':
                    image_files = temp_names
                else:
                    video_files = temp_names

        print("[INFO] Now, lets rename all photos and videos into a nice " \
            "organised format...")

        # Rename all images/videos (REAL Version)
        total_len = len(image_files) + len(video_files)
        iter = 0
        with progress('Videos checked: %s/%s', size=total_len) as prog:
            for file_type, file_supname in zip([image_files, video_files],
                    ['Image', 'Video']):
                for i, file in enumerate(file_type):
                    
                     # Update the progress bar.
                    prog.update(iter, total_len)

                    # Get directory name of file exists within
                    sup_dirname = file.parents[0].name

                    # Create new filename
                    file_new = file.parents[0] / (file_supname + '_' 
                        + sup_dirname + '_' + str(i).rjust(4,'0') 
                        + file.suffix)

                    # Rename file
                    os.rename(file, file_new)

                    iter += 1

if __name__ == '__main__':

    t_begin = time.time()
    basedir = 'C:/Users/james/pictures/Camera Roll'

    print("Welcome to Photo Processor. We are going to organise the " \
        "filenames of all your photos and videos")
    print("The directory of choice is: {dir}".format(dir=basedir))

    image_processor(basedir=basedir).rename()

    print("[INFO] All image and video files have been renamed! (In %.4fs)" % (time.time() - t_begin))
