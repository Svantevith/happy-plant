import glob
import os
import pickle

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import img_to_array

tf.compat.v1.ConfigProto().gpu_options.allow_growth = True
tf.config.run_functions_eagerly(False)


class IHappyPlant:
    LABELS_PATH = r'D:\PyCharm Professional\Projects\HappyPlant\data\models\labels\happy_plant_labels.pickle'
    MODELS_DIRECTORY = r'D:\PyCharm Professional\Projects\HappyPlant\data\models\MobileNetV2'
    DEFAULT_MODEL = os.path.join(MODELS_DIRECTORY, 'mobile_net_v2_trainable_100_accuracy_0977.hdf5')
    IMAGE_SIZE = (224, 224)
    INPUT_SHAPE = (*IMAGE_SIZE, 3)

    def __init__(
            self,
            model_buffer: str = None,
            latest: bool = False
    ):
        if model_buffer:
            self.model_buffer = model_buffer
        elif latest:
            self.model_buffer = self.get_latest_model()
        else:
            self.model_buffer = self.DEFAULT_MODEL

        print(f'[✨] Loading weights from {self.model_buffer}')
        self.model = load_model(self.model_buffer)

    @property
    def encoded_labels(self) -> dict:
        with open(self.LABELS_PATH, 'rb') as labels:
            return pickle.load(labels)

    @property
    def n_classes(self) -> int:
        return len(self.encoded_labels)

    def get_latest_model(self) -> str:
        return max(glob.iglob(os.path.join(self.MODELS_DIRECTORY, '*.hdf5')), key=os.path.getctime)

    def transform_image(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path)
        resized_image = image.resize(self.IMAGE_SIZE)
        transformed_image = img_to_array(resized_image)
        return transformed_image

    def predict_disease(self, x_test: str) -> str:
        image_array = self.transform_image(x_test)
        y_pred = np.argmax(self.model.predict(np.expand_dims(image_array, axis=0)), axis=-1)[0]
        return self.encoded_labels[y_pred].replace('_', ' ')
