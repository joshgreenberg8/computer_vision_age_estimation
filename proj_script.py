
import pandas as pd

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam

train_data = load_train('/datasets/faces/final_files/')

test_data = load_test('/datasets/faces/final_files/')

model = create_model((150, 150, 3))

train_model(model, train_data, test_data)