import matplotlib.pyplot as plt
import spektral
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.manifold import TSNE
import datetime

data = spektral.datasets.Citation("cora")
data.download()
gat = spektral.layers.GATConv
# Define Model
a_shape = data[0].a.shape
i = tf.keras.Input(shape=(a_shape[0], data.graphs[0].n_node_features,))
i2 = tf.keras.Input(shape=(*a_shape,), sparse=False)
l0 = gat(channels=10, attn_heads=10)
l1 = gat(channels=10, attn_heads=10)
x = l0([i, i2])
x = l1([x, i2])
o = tf.keras.layers.Dense(7, "softmax")(x)
model = tf.keras.models.Model([i, i2], o)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy", tf.keras.metrics.Precision()])
initial_model = tf.keras.models.clone_model(model)
# Fit model
history = model.fit([tf.expand_dims(data.graphs[0].x, 0), tf.expand_dims(data.graphs[0].a.todense(), 0)],
                    tf.expand_dims(data[0].y, 0), epochs=100)

today = datetime.datetime.now().isoformat().replace(":", "_")
model.save(f"./saved_models/{today}")
# Plot Loss/Accuracy/Precision
plt.figure()
plt.plot(history.history["loss"], label="loss")
plt.plot(history.history["accuracy"], label="accuracy")
plt.plot(history.history["precision"], label="precision")
plt.legend()
plt.savefig("./Images/loss_acc_prec.png")

# Get Predictions
preds = model.predict([tf.expand_dims(data.graphs[0].x, 0), tf.expand_dims(data.graphs[0].a.todense(), 0)])
preds = tf.squeeze(preds)
preds = tf.argmax(preds, -1)

# Plot Statistics
y = tf.argmax(data[0].y, -1)
cm = confusion_matrix(preds, y)
plt.figure()
ConfusionMatrixDisplay(cm).plot(cmap=plt.cm.Blues_r)
plt.savefig("./Images/confusion.png")

# Get intermediate Embeddings
get_embeddings = tf.keras.backend.function(model.layers[2].input, model.layers[3].output)
get_initial_embeddings = tf.keras.backend.function(initial_model.layers[2].input, initial_model.layers[3].output)
inputs = [tf.expand_dims(data.graphs[0].x, 0), tf.expand_dims(data.graphs[0].a.todense(), 0)]

initial_embs = get_initial_embeddings(inputs)
initial_embs = tf.squeeze(initial_embs)
tsne_initial = TSNE(n_components=2).fit_transform(initial_embs.numpy())
colormap = plt.cm.get_cmap("Set1")
plt.figure()
plt.scatter(tsne_initial[:, 0], tsne_initial[:, 1], c=y, cmap=colormap)
plt.savefig("./Images/initial_embeddings.png")

embs = get_embeddings(inputs)
embs = tf.squeeze(embs)
tsne = TSNE(n_components=2).fit_transform(embs.numpy())
colormap = plt.cm.get_cmap("Set1")
plt.figure()
plt.scatter(tsne[:, 0], tsne[:, 1], c=y, cmap=colormap)
plt.savefig("./Images/final_embeddings.png")
