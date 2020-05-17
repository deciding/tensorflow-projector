#import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

import glob
import os
import argparse

parser=argparse.ArgumentParser()

parser.add_argument('--log_dir', type=str, default='log/')
parser.add_argument('--spkid_npy', type=str, default='spk2spkembeddings.pickle')
parser.add_argument('--spkid_dir', type=str, default='spkid/')

args=parser.parse_args()

print("log", args.log_dir)
print("spkid_dir", args.spkid_dir)
print("spkid_npy", args.spkid_npy)
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

residual=[]
#with open('residual.txt', 'r') as f:
#    for line in f:
#        residual.append(line.strip('\n'))

# FROM NPY
if os.path.exists(args.spkid_npy):
    spk2embd=np.load(args.spkid_npy)
    X=[]
    Y_str=[]
    for Y_spk, X_spk in spk2embd.items():
        for X_utt in X_spk:
            Y_str.append(Y_spk)
            X.append(X_utt)
    X=np.array(X)
    Y_str=np.array(Y_str)
    import pdb;pdb.set_trace()
# FROM FILE
else:
    # spk list
    file_list=glob.glob(args.spkid_dir+'/*')

    # spkids
    X=np.array([np.load(fn) for fn in file_list if os.path.splitext(os.path.basename(fn))[0] not in residual])

    # labels
    Y_str=np.array([os.path.splitext(os.path.basename(fn))[0] for fn in file_list])
    Y_str=np.array([fn for fn in Y_str if fn not in residual])

print("X.shape", X.shape)
print("Y_str", Y_str.shape)

# save labels to Yspkid.tsv
np.savetxt(args.log_dir + 'Yspkid.tsv', Y_str, fmt='%s')

#exit()

embedding_var = tf.Variable(X, name='spkid')
# Format: tensorflow/tensorboard/plugins/projector/projector_config.proto
config = projector.ProjectorConfig()

# You can add multiple embeddings. Here we add only one.
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name
# Link this tensor to its metadata file (e.g. labels).
embedding.metadata_path = 'Yspkid.tsv'
#embedding.sprite.image_path = VIS_DIR + 'zalando-mnist-sprite.png'
## Specify the width and height of a single thumbnail.
#embedding.sprite.single_image_dim.extend([28, 28])

# Use the same LOG_DIR where you stored your checkpoint.
summary_writer = tf.summary.FileWriter(args.log_dir)

# The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
# read this file during startup.
projector.visualize_embeddings(summary_writer, config)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.save(sess, args.log_dir + 'model.ckpt', 0)
