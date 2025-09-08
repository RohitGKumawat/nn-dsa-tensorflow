from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers as L, models as M

def _conv_block(x, filters, kernel_size=3, strides=1, separable=False, name=None):
    if separable:
        x = L.SeparableConv2D(filters, kernel_size, strides=strides, padding="same", use_bias=False, name=None if name is None else f"{name}_sepconv")(x)
    else:
        x = L.Conv2D(filters, kernel_size, strides=strides, padding="same", use_bias=False, name=None if name is None else f"{name}_conv")(x)
    x = L.BatchNormalization(name=None if name is None else f"{name}_bn")(x)
    x = L.ReLU(name=None if name is None else f"{name}_relu")(x)
    return x

def build_simple_cnn(input_shape: Tuple[int, ...], num_classes: int) -> tf.keras.Model:
    inputs = L.Input(shape=input_shape)
    x = _conv_block(inputs, 32, 3, separable=False, name="b1")
    x = _conv_block(x, 32, 3, separable=False, name="b2")
    x = L.MaxPool2D()(x)

    x = _conv_block(x, 64, 3, separable=False, name="b3")
    x = _conv_block(x, 64, 3, separable=False, name="b4")
    x = L.MaxPool2D()(x)

    x = _conv_block(x, 128, 3, separable=False, name="b5")
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(0.3)(x)
    outputs = L.Dense(num_classes, activation="softmax")(x)
    return M.Model(inputs, outputs, name="simple_cnn")

def build_separable_cnn(input_shape: Tuple[int, ...], num_classes: int) -> tf.keras.Model:
    inputs = L.Input(shape=input_shape)
    x = _conv_block(inputs, 32, 3, separable=True, name="b1")
    x = _conv_block(x, 32, 3, separable=True, name="b2")
    x = L.MaxPool2D()(x)

    x = _conv_block(x, 64, 3, separable=True, name="b3")
    x = _conv_block(x, 64, 3, separable=True, name="b4")
    x = L.MaxPool2D()(x)

    x = _conv_block(x, 128, 3, separable=True, name="b5")
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(0.3)(x)
    outputs = L.Dense(num_classes, activation="softmax")(x)
    return M.Model(inputs, outputs, name="separable_cnn")

def build_model(model: str, input_shape: Tuple[int, ...], num_classes: int) -> tf.keras.Model:
    model = model.lower()
    if model == "simple":
        return build_simple_cnn(input_shape, num_classes)
    elif model == "separable":
        return build_separable_cnn(input_shape, num_classes)
    else:
        raise ValueError(f"Unknown model type: {model}. Choose from ['simple', 'separable']." )
