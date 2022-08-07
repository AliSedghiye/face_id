from keras.models import Model
from keras.layers import Dense, Input


from L1Dist import L1Dist
from make_embedding import embedding


class siamese_model:

  def make_siamese_model():

    embedding = embedding.make_embedding()
    input_image = Input(name='input_image', shape=(100,100,3))
    validation_image = Input(name='validation_image', shape=(100,100,3))

    # combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image), embedding(validation_image))

    classifier = Dense(1, activation='sigmoid')(distances)
    
    return Model(inputs=[input_image, validation_image], outputs=classifier, name='siameseNetwork')