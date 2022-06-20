import tensorflow as tf
import tensorflow.keras.backend as k
import tensorflow.keras.layers as l
import tensorflow.keras.losses as loss
import tensorflow.keras.activations as act
import tensorflow.keras.optimizers as optim
import tensorflow.keras.models as m
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from datetime import datetime
from spektral.datasets.citation import Citation
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import cm

KL = loss.KLDivergence(reduction='auto')


class TDistributionKernel(l.Layer):
    def __init__(self, dof=1):
        super(TDistributionKernel, self).__init__()
        self.dof = dof

    def call(self, inputs, *args, **kwargs):
        '''
        Comutes Density of student T Distribution
        :param inputs: Tuple [Z,U] where Z is of shape BxNxF and U is of shape BxCxF, where C is the number of clusters
        :param args:
        :param kwargs:
        :return: Tensor of Shape BxNxC
        '''
        Z, U = inputs
        diff = Z[..., None, :] - U[..., None, :, :]  # BxNxN
        norm = tf.norm(diff, axis=-1)  # BxN
        numerator = (1 + norm / self.dof) ** ((-self.dof + 1) / 2)
        denominator = tf.reduce_sum(numerator, axis=-1)
        qij = numerator / denominator[..., None]
        return qij


class FrequencyClusterAssignment(l.Layer):
    def __init__(self):
        super(FrequencyClusterAssignment, self).__init__()

    def call(self, inputs, *args, **kwargs):
        '''
        Computes soft cluster assignment probability
        :param inputs: Soft Cluster Assignment Probability of shape BxNxC
        :param args:
        :param kwargs:
        :return: Tensor of Shape BxNxN
        '''
        qij = inputs
        f_i = tf.expand_dims(tf.reduce_sum(qij, axis=-2), axis=-2)  # Bx1XC
        qij2 = qij ** 2
        pij_numerator = qij2 / f_i
        pij = pij_numerator / tf.reduce_sum(pij_numerator, axis=-1, keepdims=True)
        return pij


class DeepAutoencoderBlock(m.Model):
    def __init__(self, encoder: m.Model, decoder: m.Model,
                 dropout1: l.Layer = 0.2, dropout2: l.Layer = 0.2,
                 hidden_reconstruction_loss='mean_squared_error', **kwargs):
        super(DeepAutoencoderBlock, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.dropout1 = l.Dropout(dropout1)
        self.dropout2 = l.Dropout(dropout2)
        self.hidden_reconstruction_loss = hidden_reconstruction_loss
        self.n_encoder_layers = len(encoder.layers)
        self.n_decoder_layers = len(decoder.layers)
        print(self.n_encoder_layers, self.n_decoder_layers)
        assert self.n_encoder_layers + 2 == self.n_decoder_layers
        # Assumption: Dropout every other layer
        # TODO: get only dropout layer output
        self.range_decoder_layers = tf.range(0, self.n_encoder_layers-2, delta=2)
        #print(self.range_encoder_layers)
        self.reverse_range_encoder_layers = tf.reverse(self.range_decoder_layers, axis=[0])
        #print(self.reverse_range_encoder_layers)

    '''def build(self, input_shape):
        for i, j in zip(self.range_encoder_layers, self.reverse_range_decoder_layers):
            self.encoder.layers[i]._build_input_shape(input_shape)
            self.decoder.layers[i]._build_input_shape(input_shape)
            assert self.encoder.layers[i].output_shape == self.decoder.layers[i].output_shape
        tf.ensure_shape(self.decoder.output, input_shape)'''

    def call(self, inputs, training=None, mask=None):
        dropped_inputs = self.dropout1(inputs)
        encoder_out = self.encoder(dropped_inputs)
        dropped_encoder_out = self.dropout2(encoder_out)
        decoder_out = self.decoder(dropped_encoder_out)
        # reverse layer-wise reconstruction loss
        counter = 0
        for j, i in zip(self.range_decoder_layers, self.reverse_range_encoder_layers):
            print(counter)
            counter += 1
            print(i, j)
            # print(f"{self.encoder.layers[i].name, self.decoder.layers[j].name}")
            self.add_loss(loss.get(self.hidden_reconstruction_loss)(self.encoder.layers[i].output,
                                                                    self.decoder.layers[j].output)
                          )
        # input-output reconstruction loss
        self.add_loss(loss.get(self.hidden_reconstruction_loss)(inputs, decoder_out))
        return decoder_out


class Dense3D(l.Layer):
    def __init__(self, units, activation=None):
        super(Dense3D, self).__init__()
        self.units = units
        self.activation = act.get(activation)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.units))

    def call(self, inputs, *args, **kwargs):
        out = tf.einsum('BNF,FO->BNO', inputs, self.W)
        return self.activation(out)


def training_loop_clustering(inputs, encoder_model: m.Model, optimizer: optim.Optimizer, epochs=100):
    pdk = TDistributionKernel()  # Probability based on Distance Kernel
    fca = FrequencyClusterAssignment()  # Frequency Cluster Assignment
    for i in range(epochs):
        with tf.GradientTape() as tape:
            pred_clusters = encoder_model(inputs)
            q = pdk(pred_clusters)
            p = fca(q)
            loss = KL(p, q)
            print(f"Clustering Encoder Training | Epoch: {i} | Loss: {tf.reduce_sum(loss)}")
        gradients = tape.gradient(loss, encoder_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, encoder_model.trainable_variables))
    return encoder_model


def training_loop_autoencoder(inputs, autoencoder_model: DeepAutoencoderBlock,
                              optimizer: optim.Optimizer, epochs=100):
    for i in range(epochs):
        with tf.GradientTape() as tape:
            decoder_output = autoencoder_model(inputs)
            loss = autoencoder_model.losses
        print(f"Autoencoder Training | Epoch: {i} | Loss: {tf.reduce_sum(loss)}")
        gradients = tape.gradient(loss, autoencoder_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, autoencoder_model.trainable_variables))
    return autoencoder_model


def full_training_loop(inputs,
                       autoencoder_model: DeepAutoencoderBlock,
                       optimizer: str = "Adam",
                       epochs_autoencoder=50,
                       epochs_clustering=50):
    autoencoder_optimizer = optim.get(optimizer)
    clustering_encoder_optimizer = optim.get(optimizer)
    # Pre-train Autoencoder model
    autoencoder_model, loss_hist = training_loop_autoencoder(inputs, autoencoder_model=autoencoder_model,
                                                             optimizer=autoencoder_optimizer,
                                                             epochs=epochs_autoencoder)
    # Get Encoder model and discard decoder
    encoder_model = autoencoder_model.encoder
    clusters_position = "k-means++"
    for i in range(epochs_clustering):
        encoder_output = encoder_model(inputs)
        encoder_output = tf.reshape(encoder_output, (-1, encoder_output.shape[-1]))
        encoder_output_no_batch = tf.reshape(encoder_output, (-1, encoder_output.shape[-1])).numpy()
        trained_km = KMeans(n_clusters=encoder_output.shape[-1],
                            init=clusters_position,
                            n_init=15,
                            max_iter=250).fit(encoder_output_no_batch)
        # Get Cluster Position for each datapoint cluster assignment
        clusters_position = trained_km.cluster_centers_  # CxF
        cluster_assignments = trained_km.predict(encoder_output_no_batch)  # NxC
        cluster_assignments_position = clusters_position[cluster_assignments]  # NxF
        cluster_assignments_position = tf.reshape(cluster_assignments_position, encoder_output.shape)  # BxNxF
        # Train encoder to transform data points to assigned cluster position
        encoder_model = training_loop_clustering(inputs=inputs, true_clusters=cluster_assignments_position,
                                                 encoder_model=encoder_model, optimizer=clustering_encoder_optimizer,
                                                 epochs=epochs_clustering)
    return encoder_model


if __name__ == '__main__':
    cmap = cm.get_cmap("tab20")
    data = Citation("cora")
    # data.download()
    a = data.graphs[0].a.todense()
    x = data.graphs[0].x
    idxs = np.arange(x.shape[0])
    np.random.shuffle(idxs)
    batches = 50
    batch_size = 40
    idx_train = idxs[:batches * batch_size]
    idx_test = idxs[batches * batch_size:]
    x_train = x[idx_train]
    x_test = x[idx_test]
    x_train = tf.reshape(x_train, (batches, batch_size, -1))
    x_test = tf.expand_dims(x_test, 0)
    y = data.graphs[0].y
    y_train = y[idx_train]
    y_test = y[idx_test]
    y_train = tf.reshape(y_train, (batches, batch_size, -1))
    y_test = tf.expand_dims(y_test, 0)

    input_size = x.shape[-1]
    hidden_sizes_encoder = [50, 30, 10, 3]
    hidden_sizes_decoder = hidden_sizes_encoder[::-1]
    hidden_sizes_decoder.extend([input_size])
    deep_net_encoder = m.Sequential()
    for i, h in enumerate(hidden_sizes_encoder):
        if i == len(hidden_sizes_encoder):
            deep_net_encoder.add(Dense3D(h, activation="elu"))
        else:
            deep_net_encoder.add(Dense3D(h, activation=None))
        deep_net_encoder.add(l.Dropout(0.2))
    deep_net_decoder = m.Sequential()
    for i, h in enumerate(hidden_sizes_decoder):
        if i == len(hidden_sizes_decoder):
            deep_net_decoder.add(Dense3D(h, activation=None))
        else:
            deep_net_decoder.add(Dense3D(h, activation="elu"))
        deep_net_decoder.add(l.Dropout(0.2))
    autoencoder = DeepAutoencoderBlock(deep_net_encoder, deep_net_decoder)
    trained_encoder = full_training_loop(x_train, autoencoder)
    centroid_output_trained_training_set = trained_encoder(x_train).numpy().reshape(-1, 3)
    centroid_output_trained_test_set = trained_encoder(x_test).numpy().reshape(-1, 3)
    y_lab_idx_train = np.argmax(y[y_train], -1)
    y_lab_idx_test = np.argmax(y[y_test], -1)
    plt.figure()
    ax = plt.add_subplot(projection='3d')
    ax.set_title("Training Set")
    ax.scatter(centroid_output_trained_training_set, cmap(y_lab_idx_train))
    ax = plt.add_subplot(projection='3d')
    ax.set_title("Test Set")
    ax.scatter(centroid_output_trained_test_set, cmap(y_lab_idx_test))
    plt.show()
