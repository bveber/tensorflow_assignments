# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
import numpy as np
import os
import tarfile
import urllib
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
import cPickle as pickle
import time
import sys

url = 'http://yaroslavvb.com/upload/notMNIST/'

def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        filename, _ = urllib.urlretrieve(url + filename, filename)
        statinfo = os.stat(filename)
        if statinfo.st_size == expected_bytes:
            print 'Found and verified', filename
    else:
        raise Exception(
                    'Failed to verify' + filename + '. Can you get to it with a browser?')
    return filename

train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)

print 'done'

num_folders = 10

def extract(filename):
    #print 'opening %s' % filename
    ##tar = tarfile.open(filename)
    #print 'extracting'
    #tar.extractall()
    #tar.close()
    #print 'close'
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
    data_folders = [os.path.join(root, d) for d in sorted(os.listdir(root))]
    ds_store = os.path.join(filename.split('.')[0], '.DS_Store')
    if ds_store in data_folders:
        data_folders.remove(ds_store)
    if len(data_folders) != num_folders:
        raise Exception('Expected %d folders, one per class. Found %d instead.' %
                        (num_folders, len(data_folders)))
    print data_folders
    return data_folders

train_folders = extract(train_filename)
test_folders = extract(test_filename)

print 'done'
