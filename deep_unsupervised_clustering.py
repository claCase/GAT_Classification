import tensorflow as tf
import tensorflow.keras.backend as k
import tensorflow.keras.layers as l
import tensorflow.keras.models as m
import numpy
import matplotlib.pyplot as plt
import argparse
import os
from datetime import datetime


class DeepAutoencoder(m.Model):
    def __init__(self, encoder: m.Model, decoder: m.Model, **kwargs):
        super(DeepAutoencoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs, training=None, mask=None):
        encoder_out = self.encoder(inputs)
        decoder_out = self.decoder(encoder_out)
