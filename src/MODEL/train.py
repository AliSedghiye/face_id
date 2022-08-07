import os

import tensorflow as tf

from make_siamese_moel import siamese_model
from train_step import step


class train:

  def train(data, nb_epochs):

    siamese_model = siamese_model
    
    opt = tf.keras.optimizers.Adam(1e-4)

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint(opt=opt, siamese_model = siamese_model.make_siamese_model)


    # LOOP THROUGH EPOCHS
    for epoch in range(1, nb_epochs+1):
      print('\n Epoch {}/{}'.format(epoch, nb_epochs))
      progbar = tf.keras.utils.Progbar(len(data))

      # LOOP THROUGH EACH BATCH 
      for idx, batch in enumerate(data):
        # run train step here 
        step.train_step(batch)
        progbar.update(idx+1)

      # SAVE CHECKPOINTS 
      if epoch % 10 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)
