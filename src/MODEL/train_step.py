import tensorflow as tf

from make_siamese_moel import siamese_model


class step:
  @tf.function
  def train_step(batch):

    binary_cross_loss = tf.losses.BinaryCrossentropy()
    opt = tf.keras.optimizers.Adam(1e-4)

    with tf.GradientTape() as tape:
      x = batch[:2]
      y = batch[2]
      siamese_model = siamese_model
      yhat = siamese_model.make_siamese_model(x, training=True)
      loss = binary_cross_loss(y, yhat)

    print(loss)

    # calculate gradient
    grad = tape.gradient(loss, siamese_model.trainable_variables)
    # calculate updated weights and apply to siamese model 
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))

    return loss
