import tensorflow as tf
from ultralytics import YOLO

MNIST_MODEL_PATH = "./app/weights/best_model_balanced.h5"
EXTENDED_MODEL_PATH = "./app/weights/best_model_balanced.h5"
YOLO_MAIN_PATH = "./app/weights/cell_detect.pt"
YOLO_EXTRA_PATH = "./app/weights/cell_detect_extra.pt"

_mnist_model = None
_extended_model = None
_yolo_model = None
_yolo_model_extra = None

def get_mnist_model():
    global _mnist_model
    if _mnist_model is None:
        _mnist_model = tf.keras.models.load_model(MNIST_MODEL_PATH)
    return _mnist_model

def get_extended_model():
    global _extended_model
    if _extended_model is None:
        _extended_model = tf.keras.models.load_model(EXTENDED_MODEL_PATH)
    return _extended_model

def get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO(YOLO_MAIN_PATH)
    return _yolo_model

def get_yolo_model_extra():
    global _yolo_model_extra
    if _yolo_model_extra is None:
        _yolo_model_extra = YOLO(YOLO_EXTRA_PATH)
    return _yolo_model_extra
