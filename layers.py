#Custom L1 Distance Layer module

#Import Dependencies
# Needed to Load the custom model:

import tensorflow as tf
from tensorflow.keras.layers import Layer

#Custom L1 Dist layer from Jupyter
#Creating Siamese L1 Distance class
#This helps create a custom Neural Layer
class L1Dist(Layer):

    #Init method defines inheritance
    def __init__(self, **kwargs):
        super().__init__()

    #Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)
