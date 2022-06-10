import tensorflow as tf
from tensorflow.keras import backend as k
from tensorflow.keras import layers as l
from tensorflow.keras import models as m
from tensorflow.keras import initializers as init
from tensorflow.keras import regularizers as reg
from tensorflow.keras import activations as act
import numpy as np
from spektral.layers.ops import unsorted_segment_softmax
from spektral.layers.convolutional import GATConv


class GATv2Layer(l.Layer):
    def __init__(self, heads, channels,
                 concatenate_output=False,
                 activation="relu",
                 residual=False,
                 initializer=init.glorot_normal,
                 regularizer=None,
                 return_attention=False,
                 **kwargs):
        super(GATv2Layer, self).__init__(**kwargs)
        self.heads = heads
        self.channels = channels
        self.concatenate_output = concatenate_output
        self.activation = activation
        self.residual = residual
        self.initializer = initializer
        self.regularizer = regularizer
        self.return_attention = return_attention

    def build(self, input_shape):
        x, a = input_shape
        w_shape = (self.heads, x[-1], self.channels)
        a_shape = (self.heads, self.channels)
        self.W_self = self.add_weight(name=f"kern_features_self",
                                      shape=w_shape,
                                      initializer=self.initializer,
                                      regularizer=self.regularizer,
                                      trainable=True
                                      )
        self.W_ngb = self.add_weight(name=f"kern_features_ngb",
                                     shape=w_shape,
                                     initializer=self.initializer,
                                     regularizer=self.regularizer,
                                     trainable=True
                                     )
        self.attn = self.add_weight(name=f"kern_attention",
                                    shape=a_shape,
                                    initializer=self.initializer,
                                    regularizer=self.regularizer,
                                    trainable=True
                                    )

    def call(self, inputs, training=None, mask=None):
        '''
        When the adjacency matrix is a Sparse Tensor batch size is not supported
        :param inputs: Tuple of features NxF and Sparse Adjacency Matrix NxN
        :param training: Whether in training mode
        :param mask: Not Used
        :return: Updated Features of shape Nx(HF) or NF
        '''
        x, a = inputs
        N = x.shape[-2]
        if isinstance(a, tf.sparse.SparseTensor):
            i, j = a.indices[:, 0], a.indices[:, 1]
            x_i = tf.gather(x, i)
            x_i_prime = tf.einsum("EF,HFO->EHO", x_i, self.W_self)
            x_j = tf.gather(x, j)
            x_j_prime = tf.einsum("EF,HFO->EHO", x_j, self.W_ngb)
            x_ij_prime = x_i_prime + x_j_prime
            x_ij_prime = act.get(self.activation)(x_ij_prime)
            a_ij = tf.einsum("EHO,HO->EH", x_ij_prime, self.attn)
            a_soft = unsorted_segment_softmax(a_ij, j, N)
            out = a_soft[..., None] * x_i[:, None]  # EH
            out = tf.math.unsorted_segment_sum(out, j, N)  # NHF
        else:
            raise NotImplemented(f"Adjacency Matrix must be a Sparse Tensor, but is {type(a)}")
        if self.concatenate_output:
            out = tf.reshape(out, (N, -1))
        else:
            out = tf.math.reduce_mean(out, -2)
        if self.return_attention:
            return out, a_soft
        else:
            return out


class GATv2Model(m.Model):
    def __init__(self, channels=[10,10,7], hidden_sizes=[10,10,7], heads=[10,10,1], activation="elu", *args, **kwargs):
        super(GATv2Model, self).__init__(**kwargs)
        self.channels = channels
        self.hidden_sizes = hidden_sizes
        self.heads = heads
        assert len(channels) == len(heads) == len(hidden_sizes)
        self.n_layers = len(hidden_sizes)
        self.activation = activation
        self.gat_layers = [GATv2Layer(heads=hh, channels=c, activation=activation, *args) for c, hh in
                           zip(self.channels, self.heads)]
        self.dense_layers = [l.Dense(h, activation) for h in self.hidden_sizes]

    def call(self, inputs, training=None, mask=None):
        x, a = inputs
        for d, g in zip(self.dense_layers, self.gat_layers):
            x = d(x)
            x = g([x, a])
        return x


class GATModel(m.Model):
    def __init__(self, channels, hidden_sizes, heads, activation="elu", *args, **kwargs):
        super(GATModel, self).__init__(**kwargs)
        self.channels = channels
        self.hidden_sizes = hidden_sizes
        self.heads = heads
        assert len(channels) == len(heads) == len(hidden_sizes)
        self.n_layers = len(hidden_sizes)
        self.activation = activation
        self.gat_layers = [GATConv(attn_heads=hh, channels=c, activation=activation, *args) for c, hh in
                           zip(self.channels, self.heads)]
        self.dense_layers = [l.Dense(h, activation) for h in self.hidden_sizes]

    def call(self, inputs, training=None, mask=None):
        x, a = inputs
        for d, g in zip(self.dense_layers, self.gat_layers):
            x = d(x)
            x = g([x, a])
        return x


if __name__ == "__main__":
    from spektral import datasets
    import matplotlib.pyplot as plt

    data = datasets.Citation("cora")
    data.download()
    a = tf.sparse.from_dense(data.graphs[0].a.todense())
    x = data.graphs[0].x
    y = data.graphs[0].y
    i = l.Input(shape=(x.shape[-1]))
    o = l.Dense(10, "elu")(i)
    o = l.Dense(10, "elu")(o)
    o = l.Dense(7, "elu")(o)
    dense_model = m.Model(i,o)

    gat = GATModel(channels=[10, 10, 7], hidden_sizes=[15, 15, 7], heads=[10, 10, 1])
    gat2 = GATv2Model(channels=[10, 10, 7], hidden_sizes=[15, 15, 7], heads=[10, 10, 1])
    loss_hist, loss2_hist, loss3_hist = [], [], []
    optimizer = tf.keras.optimizers.Adam()
    optimizer2 = tf.keras.optimizers.Adam()
    optimizer3 = tf.keras.optimizers.Adam()
    for i in range(400):
        print(f"Epoch: {i}")
        with tf.GradientTape() as tape:
            out = gat([x, a])
            loss = tf.keras.losses.categorical_crossentropy(y, out, from_logits=True)
            loss = tf.reduce_mean(loss)
            loss_hist.append(loss)
        grads = tape.gradient(loss, gat.trainable_weights)
        optimizer.apply_gradients(zip(grads, gat.trainable_weights))
        with tf.GradientTape() as tape:
            out2 = gat2([x, a])
            loss2 = tf.keras.losses.categorical_crossentropy(y, out2, from_logits=True)
            loss2 = tf.reduce_mean(loss2)
            loss2_hist.append(loss2)
        grads2 = tape.gradient(loss2, gat2.trainable_weights)
        optimizer2.apply_gradients(zip(grads2, gat2.trainable_weights))
        with tf.GradientTape() as tape:
            out3 = dense_model(x)
            loss3 = tf.keras.losses.categorical_crossentropy(y, out3, from_logits=True)
            loss3 = tf.reduce_mean(loss3)
            loss3_hist.append(loss3)
        grads3 = tape.gradient(loss3, dense_model.trainable_weights)
        optimizer3.apply_gradients(zip(grads3, dense_model.trainable_weights))

    plt.plot(loss_hist, label="GAT Loss")
    plt.plot(loss2_hist, label="GATv2 Loss")
    plt.plot(loss3_hist, label="Dense Loss")
    plt.legend()
    plt.show()