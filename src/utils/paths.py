# -------------------------------------------------------------------
# File:  path.py
# Author: boren li <borenli@cea-igp.ac.cn>
# Created: 2019-6-14
# ------------------------------------------------------------------#
"""
Path Uilts
"""
# import the necessary packages
import os


def list_files_path(basePath, validExts=None, contains=None):
    """
    List all files in the specified path
    
    :param basePath: The specified path
    :param validExts: The file extention
    :param contains: The contains string whether is contained by the filename  
    """
    # return the set of files that are valid
    return list(list_files(basePath, validExts=validExts, contains=contains))


def list_files(basePath, validExts=None, contains=None):
    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()

            # check to see if the file is an image and should be processed
            if validExts is None or ext.endswith(validExts):
                # construct the path to the image and yield it
                filePath = os.path.join(rootDir, filename)
                yield filePath
