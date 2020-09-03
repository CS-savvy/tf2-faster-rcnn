from network import frcnn_resnet
from pathlib import Path
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input
from tensorflow.keras import Model

this_dir = Path.cwd()

image_shape = (224, 224, 3)



image_input = Input(shape= image_shape)
base_layers = frcnn_resnet.nn_base(image_input, trainable=True)

base_model = Model(image_input, base_layers)

print(base_model.summary())

plot_model(base_model, "base_model.png", show_shapes=True, show_layer_names=True)