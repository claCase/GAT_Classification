from gatv2 import GATv2Model
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
import datetime


def sparse_diag(A: tf.sparse.SparseTensor):
    diag = tf.sparse.reduce_sum(A, -1)
    i = tf.range(A.shape[-1], dtype=tf.int64)[:, None]
    ii = tf.concat([i, i], -1)
    return tf.sparse.SparseTensor(ii, diag, A.shape)


class FAGCNLayer(l.Layer):
    def __init__(self, dropout, **kwargs):
        super(FAGCNLayer, self).__init__(**kwargs)
        self.dropout = l.Dropout(dropout)

    def build(self, input_shape):
        x, a = input_shape
        self.kernel = self.add_weight(shape=(2 * x[-1],1), name="kernel", initializer="glorot_normal")

    def call(self, inputs, *args, **kwargs):
        x, a = inputs
        if isinstance(a, tf.sparse.SparseTensor):
            tf.assert_rank(a, 2)
            N = tf.shape(x, out_type=a.indices.dtype)[-2]
            diag = sparse_diag(a)
            sqrt_diag = tf.math.sqrt(diag.values)[:, None]  # Nx1
            i, j = a.indices[:, 0], a.indices[:, 1]
            x_i = tf.gather(x, i)  # ExF
            x_j = tf.gather(x, j)  # ExF
            x_ij = tf.concat([x_i, x_j], -1)  # Ex2F
            alpha = tf.nn.tanh(x_ij @ self.kernel)  # Ex1
            di = tf.gather(sqrt_diag, i)  # Ex1
            dj = tf.gather(sqrt_diag, j)  # Ex1
            dij = tf.where(di * dj == 0.0, k.epsilon(), di * dj)  # Ex1
            g_coeff = alpha / dij  # Ex1
            g_coeff = self.dropout(g_coeff)
            x_i_weighted = x_i * g_coeff  # ExF
            x_i_summed = tf.math.unsorted_segment_sum(x_i_weighted, j, N)  # NxF
            #x_j_prime = self.epsilon * x + x_i_summed  # NxF
        return x_i_summed


class FAGCNModel(m.Model):
    def __init__(self, hidden_sizes, n_layers, dropout, activation="elu", epsylon=0.2, **kwargs):
        super(FAGCNModel, self).__init__(**kwargs)
        self.hidden_sizes = hidden_sizes
        assert len(hidden_sizes) == 2
        self.n_layers = n_layers
        self.activation = activation
        self.epsylon = epsylon
        self.dropout = l.Dropout(dropout)
        self.graph_layers = [FAGCNLayer(dropout) for _ in range(n_layers)]
        self.dense_layers = [l.Dense(h, activation) for h in hidden_sizes]

    def call(self, inputs, training=None, mask=None):
        x, a = inputs
        x = self.dense_layers[0](x)
        self.dropout(x)
        raw = x
        for g in self.graph_layers:
            x_prime = g([x, a])
            x = self.epsylon * raw + x_prime
        x = self.dense_layers[1](x)
        return x


if __name__ == "__main__":
    from spektral import datasets
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    sparse = args.sparse
    epochs = args.epochs

    data = datasets.Citation("cora")
    data.download()
    a = data.graphs[0].a.todense()
    x = data.graphs[0].x
    y = data.graphs[0].y
    if sparse:
        a = tf.sparse.from_dense(a)
    else:
        N = 500
        a = a[None, :N, :N]
        x = x[None, :N]
        y = y[None, :N]

    fgcn = FAGCNModel(hidden_sizes=[25, 7], n_layers=4, dropout=0.2)
    gat2 = GATv2Model(channels=[25, 15, 7], hidden_sizes=[25, 15, 7], heads=[1, 1, 1])
    loss_hist, loss2_hist = [], []
    optimizer = tf.keras.optimizers.Adam()
    optimizer2 = tf.keras.optimizers.Adam()
    for i in range(400):
        print(f"Epoch: {i}")
        with tf.GradientTape() as tape:
            out = fgcn([x, a])
            loss = tf.keras.losses.categorical_crossentropy(y, out, from_logits=True)
            loss = tf.reduce_mean(loss)
            loss_hist.append(loss)
        grads = tape.gradient(loss, fgcn.trainable_weights)
        optimizer.apply_gradients(zip(grads, fgcn.trainable_weights))
        with tf.GradientTape() as tape:
            out2 = gat2([x, a])
            loss2 = tf.keras.losses.categorical_crossentropy(y, out2, from_logits=True)
            loss2 = tf.reduce_mean(loss2)
            loss2_hist.append(loss2)
        grads2 = tape.gradient(loss2, gat2.trainable_weights)
        optimizer2.apply_gradients(zip(grads2, gat2.trainable_weights))

    plt.plot(loss_hist, label="FGCN Loss")
    plt.plot(loss2_hist, label="GATv2 Loss")
    plt.legend()
    plt.show()