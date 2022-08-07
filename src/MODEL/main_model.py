import os

import tensorflow as tf

from capture import camera
from preprocess_twin import preprocess_twin
from train import train

# define directory path
pos_data = os.path.join('src/MODEL/data', 'posetive')
neg_data = os.path.join('src/MODEL/data', 'negative')
anch_data = os. path.join('src/MODEL/data', 'anchor')

# create dirs
os.makedirs(pos_data)
os.makedirs(neg_data)
os.makedirs(anch_data)

# unzip data file
# run this code in comand line to unzip negative data 
# tar -xf 'file path'
    #! note : This file is placed in the attachment named 'lfw.tgz'

# move data to negative data file
for directory in os.listdir('src/MODEL/lfw'):
    for file in os.listdir(os.path.join('src/MODEL/lfw', directory)):
        ex_path = os.path.join('src/MODEL/lfw', directory, file)
        next_path = os.path.join(neg_data, file)
        os.replace(ex_path, next_path)

camera.collect(anch_data, pos_data)

anchor = tf.data.Dataset.list_files(anch_data+'/*.jpg').take(500)
posetive = tf.data.Dataset.list_files(pos_data+'/*.jpg').take(500)
negative = tf.data.Dataset.list_files(neg_data+'/*.jpg').take(500)

posetive = tf.data.Dataset.zip((anchor, posetive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negative = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = posetive.concatenate(negative)

# build dataloder pipeline
data = data.map(preprocess_twin().preprocess_twin())
data = data.cache()
data = data.shuffle(buffer_size=1024)

# train data
train_data = data.take(round(len(data)*0.7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)


# test data 
# if you want test with this data 
test_data = data.skip(round(len(data)*0.7))
test_data = test_data.take(round(len(data)*0.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)


EPOCH = 50
train.train(train_data, EPOCH)
