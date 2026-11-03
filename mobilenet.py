"""Keras MobileNetV2; PyTorch backend reuses the installed CPU runtime."""
import os
from pathlib import Path
os.environ.setdefault('KERAS_BACKEND', 'torch')
os.environ.setdefault('KERAS_HOME', str(Path(__file__).resolve().parent.parent/'keras-cache'))
import keras
import numpy as np
from PIL import ImageOps

IMAGE_SIZE = 224
CLASSES = ['fake', 'real']

def image_array(image):
    image = ImageOps.pad(ImageOps.exif_transpose(image).convert('RGB'), (IMAGE_SIZE, IMAGE_SIZE))
    return np.asarray(image, dtype=np.uint8).copy()

def compile_model(model, learning_rate):
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

def build_model(weights='imagenet'):
    base_model = keras.applications.MobileNetV2(
        weights=weights, include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    base_model.trainable = False
    inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name='image')
    # Equivalent to MobileNetV2 preprocess_input; persisted inside the checkpoint.
    x = keras.layers.Rescaling(1./127.5, offset=-1., name='imagenet_preprocessing')(inputs)
    # training=False keeps moving BatchNormalization statistics fixed in both stages.
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(.3)(x)
    outputs = keras.layers.Dense(2, activation='softmax', name='gold_classes')(x)
    model = keras.Model(inputs, outputs, name='gold_mobilenet_v2')
    compile_model(model, 1e-3)
    return model, base_model

def enable_fine_tuning(model, base_model):
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    for layer in base_model.layers[-30:]:
        layer.trainable = not isinstance(layer, keras.layers.BatchNormalization)
    # Explicitly preserve every BN layer, including moving means and variances.
    for layer in base_model.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
    # A fresh optimizer/compile is essential after changing trainable flags.
    compile_model(model, 1e-5)
    return [{'name': layer.name, 'trainable': layer.trainable,
             'type': type(layer).__name__} for layer in base_model.layers]

def predict_scores(model, image):
    return np.asarray(keras.ops.convert_to_numpy(model(image_array(image)[None], training=False)))[0]
