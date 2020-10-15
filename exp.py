from network import frcnn_resnet
from pathlib import Path
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input
from tensorflow.keras import Model

this_dir = Path.cwd()

image_shape = (512, 512, 3)



image_input = Input(shape= image_shape)
base_layers = frcnn_resnet.nn_base(image_input, trainable=True)

rpn = frcnn_resnet.rpn(base_layers, 9)

base_model = Model(image_input, base_layers)
model_rpn = Model(image_input, rpn[:2])

print(model_rpn.summary())

#plot_model(model_rpn, "model_rpn.png", show_shapes=True, show_layer_names=True)