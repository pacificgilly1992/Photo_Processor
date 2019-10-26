__title__ = 'Image and Video Organiser'
__description__ = 'Fixes the naming of all videos and images within a directory\
                    or folder by renaming all images and videos using the\
                    subdirectory and age of file as a basis'
__author__ = 'James Gilmore'
__date__ = '21/02/2019'
__version__ = '0.2'

import os
import sys
import glob

import numpy as np

from datetime import datetime

from PIL import Image

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
        self.image_formats = ['.jpg', '.JPG', '.tiff', '.bmp', '.png', '.PNG', '.gif', '.GIF', 'jpeg']
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

    def _findfiles(self, basedir, images, videos):
        """
        Finds all images and videos within basedir
        """

        if images is True:

            # Locate all files in selected directory
            files = glob.iglob(basedir + '**/*', recursive=True)

            # Subset image files for specific extension (e.g. .jpg, .JPG, 
            # .png, .tiff, .bmp)
            image_files = self._subset_files(files, self.image_formats)

        else:

            image_files = None

        if videos is True:

            # Locate all files in selected directory
            files = glob.iglob(basedir + '**/*', recursive=True)

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
            if file[-4:] in extensions:
                files.append(file)

        # Return a subsetted filelist
        return files

    def _get_creation(self, filelist, filetypes):
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

        if filetypes == 'images':
            creation_date = []
            for file in filelist:
                try:
                    creation_date.append(Image.open(file)._getexif()[36867])
                except (KeyError, TypeError, AttributeError):
                    # Only works on Windows
                    creation_date.append(
                        datetime.fromtimestamp(
                            os.path.getmtime(file)).strftime('%Y:%m:%d %H:%M:%S'))

        elif filetypes == 'videos':
            creation_date = []
            for file in filelist:
                creation_date.append(
                            datetime.fromtimestamp(
                                os.path.getmtime(file)).strftime('%Y:%m:%d %H:%M:%S'))

        return creation_date

    def rename(self):
        """
        Method used to rename all files within basedir. Follows the 
        filename format,
        
        image_<sub_directory>_imagenum.<original_file_format>
        """

        # Convert self.image_files and self.video_files to numpy
        if self.image_files is not None:
            self.image_files = np.array(self.image_files, dtype=str)

        if self.video_files is not None:
            self.video_files = np.array(self.video_files, dtype=str)

        # Get creation dates for all image files
        image_creation_date = self._get_creation(self.image_files, filetypes='images')

        # Sort image_files by date
        mask = np.argsort(image_creation_date, kind='mergesort')
        self.image_files = self.image_files[mask]

        # Get creation dates for all video files
        video_creation_date = self._get_creation(self.video_files, filetypes='videos')

        # Sort image_files by date
        mask = np.argsort(video_creation_date, kind='mergesort')
        self.video_files = self.video_files[mask]

        # print("image_creation_date", image_creation_date)
        # print("video_creation_date", video_creation_date)

        # Rename all images/videos (FAKE Version)
        for file_type, file_supname in zip([self.image_files, self.video_files],
            ['Image', 'Video']):
            temp_names = []
            for i, file in enumerate(file_type):

                # Get directory name of file exists within
                sup_dirname = os.path.basename(os.path.dirname(file))

                # Create new filename
                temp_names.append(os.path.dirname(file) + '/' + file_supname + '_' + sup_dirname + '_' + str(i).rjust(4,'0') + 'temp' + file[-4:].lower())

                # Rename file
                os.rename(file, temp_names[-1])

            if file_supname == 'Image':
                image_files = temp_names
            else:
                video_files = temp_names

        # Rename all images/videos (REAL Version)
        for file_type, file_supname in zip([image_files, video_files],
            ['Image', 'Video']):
            for i, file in enumerate(file_type):

                # Get directory name of file exists within
                sup_dirname = os.path.basename(os.path.dirname(file))

                # Create new filename
                file_new = os.path.dirname(file) + '/' + file_supname + '_' + sup_dirname + '_' + str(i).rjust(4,'0') + file[-4:].lower()

                # Rename file
                os.rename(file, file_new)


        return

if __name__ == '__main__':

    basedir = 'C:/Users/james/pictures/Camera Roll'

    image_processor(basedir=basedir).rename()

    print("[INFO] All image and video files have been renamed!")
