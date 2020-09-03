from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications import ResNet50


resnet = ResNet50(input_shape=(224, 224, 3))
plot_model(resnet, "complete_resnet.png", show_shapes=True, show_layer_names=True)
print(resnet.summary())