#!/usr/bin/env python3

import os
import glob
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import fid
import tensorflow as tf
import h5py


########
# PATHS
########

data_path = 'celeba_img.h5py' # set path to training set images
output_path = 'fid_stats.npz' # path for where to store the statistics
# if you have downloaded and extracted
#   http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
# set this path to the directory where the extracted files are, otherwise
# just set it to None and the script will later download the files for you
inception_path = None
print("check for inception model..", end=" ", flush=True)
inception_path = fid.check_or_download_inception(inception_path) # download inception if necessary
print("ok")

# loads all images into memory (this might require a lot of RAM!)
print("load images..", end=" " , flush=True)
# image_list = glob.glob(os.path.join(data_path, '*.jpg'))
filedata = h5py.File(data_path,'r')
filedata = filedata['celeba']
images = filedata[:50000]

print("%d images found and loaded" % len(images))

print("create inception graph..", end=" ", flush=True)
fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
print("ok")

print("calculte FID stats..", end=" ", flush=True)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    mu, sigma = fid.calculate_activation_statistics(images, sess, batch_size=100)
    np.savez_compressed(output_path, mu=mu, sigma=sigma)
print("finished")
