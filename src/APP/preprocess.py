import tensorflow as tf

class preprocess:

  def __init__(self, file_path) :
    self.file_path = file_path

  def preprocess(self):
    byte_img = tf.io.read_file(self.file_path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (100,100))
    img = img / 255.0

    return img