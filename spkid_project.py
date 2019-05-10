import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

import glob
import os
import argparse

parser=argparse.ArgumentParser()

parser.add_argument('--vis_dir', type=str, default='/home/zhangzining/Documents/work/speaker/tensorflow-projector/visualization')
parser.add_argument('--log_dir', type=str, default='log/')
parser.add_argument('--spkid_dir', type=str, default='spkid/')

args=parser.parse_args()

print("vis", args.vis_dir)
print("log", args.log_dir)
print("spkid_dir", args.spkid_dir)
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

residual=[]
with open('residual.txt', 'r') as f:
    for line in f:
        residual.append(line.strip('\n'))

file_list=glob.glob(args.spkid_dir+'/*')

#X,Y
X=np.array([np.load(fn) for fn in file_list if os.path.splitext(os.path.basename(fn))[0] not in residual])
#X=np.array([np.load(fn) for fn in file_list])

#Y_str
Y_str=np.array([os.path.splitext(os.path.basename(fn))[0] for fn in file_list])
Y_str=np.array([fn for fn in Y_str if fn not in residual])
#print("X[0]", X[0])
print("X.shape", X.shape)
#print("Y_str", Y_str)
print("Y_str", Y_str.shape)

np.savetxt(args.vis_dir + '/Yspkid.tsv', Y_str, fmt='%s')

#exit()

embedding_var = tf.Variable(X, name='spkid')
# Format: tensorflow/tensorboard/plugins/projector/projector_config.proto
config = projector.ProjectorConfig()

# You can add multiple embeddings. Here we add only one.
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name
# Link this tensor to its metadata file (e.g. labels).
embedding.metadata_path = args.vis_dir + '/Yspkid.tsv'
#embedding.sprite.image_path = VIS_DIR + 'zalando-mnist-sprite.png'
## Specify the width and height of a single thumbnail.
#embedding.sprite.single_image_dim.extend([28, 28])

# Use the same LOG_DIR where you stored your checkpoint.
summary_writer = tf.summary.FileWriter(args.log_dir + '/visualization')

# The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
# read this file during startup.
projector.visualize_embeddings(summary_writer, config)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.save(sess, args.log_dir + '/visualization/model.ckpt', 0)
