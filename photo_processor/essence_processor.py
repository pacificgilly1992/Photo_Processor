__title__ = "Image and Video Organiser"
__description__ = "Fixes the naming of all videos and images within a directory\
                    or folder by renaming all images and videos using the\
                    subdirectory and age of file as a basis"
__author__ = "James Gilmore"
__date__ = "27/10/2019"
__version__ = "0.6"

import argparse
import hashlib
import os
import random
import string
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from gilly_utilities import progress
from PIL import Image

"""
New features to work on:
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

    # Define the image and video formats to check for.
    image_formats = [
        ".jpg",
        ".jpeg",
        ".tiff",
        ".bmp",
        ".png",
        ".gif",
        ".heic",
        ".heif",
        ".heifs",
        ".hvif",
        ".avci",
        ".avcs",
        ".svg",
        ".dxf",
    ]
    video_formats = [".mp4", ".mov", ".avi", ".hevc"]

    def __init__(self, basedir, images=True, videos=True):
        """
        Finds all files within basedir and renames all files within all
        subdirectories with the filename format,

        image_<sub_directory>_<imagenum>.<original_file_format>

        The imagenum is sequential based on only the images in each
        subdirectory and is invariant of image file type. The sequential
        behaviour is based on the date the picture was created.

        Parameters
        ----------
        basedir : str,
            The directory location you want to organize all images into.
            The directory location can be the absolute or relative path.
        images : boolean, optional, default = True
            Specify whether to process images within the subdirectories.
        videos : boolean, optional, default = True
            Specify whether to process videos within the subdirectories.
        """

        # Error checking
        if not isinstance(basedir, str):
            raise ValueError(
                "'basedir' must be a string defining the \
                                location of all images and videos to \
                                process"
            )

        if not isinstance(images, bool):
            raise ValueError("'images' must be either True or False")

        if not isinstance(videos, bool):
            raise ValueError("'videos' must be either True or False")

        if not Path(basedir).is_dir():
            raise IOError("'basedir' must link to an existing directory")

        # Ensure correct format of basedir
        self.basedir = os.path.abspath(basedir)

        # Find all images and videos within basedir
        self.image_files, self.video_files = self._findfiles(
            basedir=self.basedir, images=images, videos=videos
        )

        print(
            "[INFO] Processing {photo} photos and {video} videos".format(
                photo=self.image_files.size, video=self.video_files.size
            )
        )

    @staticmethod
    def _sha256sum(fpath):
        """
        Calculate the sha256 hash of a file quickly. This is used to
        ensure the Baton report downloaded into '/tmp/' is the same one
        we are uploading to Mediator.
        """

        h = hashlib.sha256()

        with open(fpath, "rb") as file:
            while True:
                # Reading is buffered, so we can read smaller chunks.
                chunk = file.read(h.block_size)
                if not chunk:
                    break
                h.update(chunk)

        return h.hexdigest()

    def _findfiles(self, basedir, images=False, videos=False):
        """
        Finds all images and videos within basedir
        """

        # Initalise the Path class
        path = Path(basedir)

        # Locate all files in selected directory
        files = list(path.glob("**/*"))

        # Subset image files for specific extension (e.g. .jpg, .JPG, .png,
        # .tiff, .bmp)
        if images:
            image_files = self._subset_files(files, self.image_formats)

        else:
            image_files = np.array([])

        # Subset video files for specific extension (e.g. .avi, .mp4, .mov)
        if videos:
            video_files = self._subset_files(files, self.video_formats)

        else:
            video_files = np.array([])

        return image_files, video_files

    def _subset_files(self, filelist, extensions):
        """
        Subset filelist with extensions specified
        """

        # Search in each file if the extension matches the end of the file.
        return np.array(
            [file for file in filelist if file.suffix.lower() in extensions], dtype=str
        )

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

        if file_type == "images":
            creation_date = []
            with progress("Photos checked: %s/%s", size=len(file_list)) as prog:
                for i, file in enumerate(file_list):

                    # Update the progress bar.
                    prog.update(i, len(file_list))

                    try:
                        creation_date.append(Image.open(file)._getexif()[36867])
                    except (KeyError, TypeError, AttributeError):
                        # Only works on Windows
                        creation_date.append(
                            datetime.fromtimestamp(os.path.getmtime(file)).strftime(
                                "%Y:%m:%d %H:%M:%S"
                            )
                        )

        elif file_type == "videos":
            creation_date = []
            with progress("Videos checked: %s/%s", size=len(file_list)) as prog:
                for i, file in enumerate(file_list):

                    # Update the progress bar.
                    prog.update(i, len(file_list))

                    creation_date.append(
                        datetime.fromtimestamp(os.path.getmtime(file)).strftime(
                            "%Y:%m:%d %H:%M:%S"
                        )
                    )

        return creation_date

    def _generate_temp_name(self, N=8):
        """
        Generate a tempory name by creating a random set of characters.

        :param N : int, optional
            Specify the number of characters you want the temporary name
            to be.
        """

        return "".join(
            random.SystemRandom().choice(string.ascii_uppercase + string.digits)
            for _ in range(N)
        )

    def _remove_duplicates(self):
        """
        Checks for duplicate files using hashing algorithm.
        """

        # Hash all files
        total_len = len(self.image_files) + len(self.video_files)
        iter = 0
        hash_files = {"Image": [], "Video": []}
        with progress("Files hashed: %s/%s", size=total_len) as prog:
            for file_type, file_supname in zip(
                [self.image_files, self.video_files], ["Image", "Video"]
            ):
                for file in file_type:

                    # Update the progress bar.
                    prog.update(iter, total_len)

                    hash_files[file_supname].append(self._sha256sum(file))

                    iter += 1

                # Convert hash_files into numpy arrays for complex indexing.
                hash_files[file_supname] = np.array(hash_files[file_supname], dtype=str)

        print("[INFO] Removing all but one duplicate images...")

        # Check for duplicates
        for file_type, file_supname in zip(
            [self.image_files, self.video_files], ["Image", "Video"]
        ):

            # Convert hash_files into numpy arrays for complex indexing.
            file_type = np.array(file_type, dtype=str)

            # Work out the non_unique hashes
            unique, counts = np.unique(hash_files[file_supname], return_counts=True)
            not_unique = unique[counts > 1]

            print(
                "[INFO] Number of duplicate {type} found are: {num}".format(
                    type=file_supname.lower() + "s", num=not_unique.size
                )
            )

            # Loop over each duplicate hash and find the files that match that
            # hash value and remove all but one of the files.
            for hash in not_unique:
                for file in file_type[hash_files[file_supname] == hash][1:]:
                    print("[INFO] Removing file: {file}".format(file=file))
                    os.remove(file)

        # Now need to recheck the files
        self.image_files, self.video_files = self._findfiles(
            basedir=self.basedir, images=True, videos=True
        )

    def rename(self, remove_duplicates=False):
        """
        Method used to rename all files within basedir. Follows the
        filename format,

        image_<sub_directory>_imagenum.<original_file_format>

        :param remove_duplicates : bool, optional
            Remove the duplicate images and videos. This is done by hashing
            all the images and videos (SHA256) and finding clashes.
        """

        print("[INFO] Renaming your photos and videos.")

        # Check for duplicate files if requested.
        if remove_duplicates:
            print("[INFO] Checking for duplicate image and video files")
            self._remove_duplicates()

        print("[INFO] Getting the creation dates of all your photos...")

        # Get creation dates for all image files
        image_creation_date = self._get_creation(self.image_files, file_type="images")

        # Sort image_files by date
        mask = np.argsort(image_creation_date, kind="mergesort")
        self.image_files = self.image_files[mask]

        print("[INFO] Getting the creation dates of all your videos...")

        # Get creation dates for all video files
        video_creation_date = self._get_creation(self.video_files, file_type="videos")

        # Sort image_files by date
        mask = np.argsort(video_creation_date, kind="mergesort")
        self.video_files = self.video_files[mask]

        print(
            "[INFO] Creating tempory names for you photos and videos. "
            "This avoids errors if you run this program multiple times."
        )

        # Rename all images/videos (FAKE Version)
        total_len = len(self.image_files) + len(self.video_files)
        iter = 0
        with progress("Videos checked: %s/%s", size=total_len) as prog:
            for file_type, file_supname in zip(
                [self.image_files, self.video_files], ["Image", "Video"]
            ):
                temp_names = []

                for i, file in enumerate(file_type):

                    # Update the progress bar.
                    prog.update(iter, total_len)

                    # Create new filename
                    temp_names.append(
                        Path(file).parents[0]
                        / (self._generate_temp_name() + Path(file).suffix)
                    )

                    # Rename file
                    os.rename(file, temp_names[-1])

                    iter += 1

                if file_supname == "Image":
                    image_files = temp_names
                else:
                    video_files = temp_names

        print(
            "[INFO] Now, lets rename all photos and videos into a nice "
            "organised format..."
        )

        # Rename all images/videos (REAL Version)
        total_len = len(image_files) + len(video_files)
        iter = 0
        dir_iter = defaultdict(int)
        with progress("Videos checked: %s/%s", size=total_len) as prog:
            for file_type, file_supname in zip(
                [image_files, video_files], ["Image", "Video"]
            ):
                for i, file in enumerate(file_type):

                    # Update the progress bar.
                    prog.update(iter, total_len)

                    # Get directory name of file exists within
                    sup_dirname = file.parents[0].name

                    # Get the image number for the specific directory. We use 4 significant figures to populate the
                    # image name space to ensure consistent filename width.
                    image_num = str(dir_iter[sup_dirname]).rjust(4, "0")

                    # Create new filename
                    file_new = f"{file.parents[0]}/{file_supname}_{sup_dirname.replace(' ', '_')}_{image_num}{file.suffix}"

                    # Rename file
                    os.rename(file, file_new)

                    iter += 1
                    dir_iter[sup_dirname] += 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Rename all image and video files within a chosen " "directory"
    )

    parser.add_argument(
        "--path",
        action="store",
        dest="path",
        type=str,
        help="The absolute or relative path pointing to the directory that "
        " you want to rename all image and video files for.",
        required=True,
    )

    parser.add_argument(
        "--images",
        action="store_true",
        dest="images",
        help="Specify whether you want to process the images in the directory",
    )

    parser.add_argument(
        "--videos",
        action="store_true",
        dest="videos",
        help="Specify whether you want to process the videos in the directory",
    )

    parser.add_argument(
        "--remove_duplicates",
        action="store_true",
        dest="remove_duplicates",
        help="Specify whether you want remove any duplicate image or video. "
        "This method will calculate the sha256 hash of the files and "
        "cross reference all files to find any duplicates. Therefore "
        "this method is safe and will NOT mistakenly delete non "
        "duplicate files.",
    )

    arg = parser.parse_args()

    if not arg.images and not arg.videos:
        raise ValueError(
            "Must specify either --images or --videos (or both) " "to rename."
        )

    t_begin = time.time()

    print(
        "Welcome to Photo Processor. We are going to organise the "
        "filenames of all your photos and videos"
    )
    print("The directory of choice is: {dir}".format(dir=arg.path))

    # Initialise the image_processor.
    content = image_processor(basedir=arg.path, images=arg.images, videos=arg.videos)

    # Rename the videos and images.
    content.rename(remove_duplicates=arg.remove_duplicates)

    print(
        "[INFO] All image and video files have been renamed! (In "
        "{:.4f}s)".format(time.time() - t_begin)
    )
