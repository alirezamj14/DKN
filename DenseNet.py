import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, UpSampling2D, Flatten, BatchNormalization, Dense, Dropout, GlobalAveragePooling2D
from MyFunctions import *

Myactivation='softmax'
Myloss='categorical_crossentropy'

parameters_path = "./parameters/"
data = 'mnist'
output_dic = {}

# Model / data parameters
p = 0.01
K_shots = 1
num_classes = 10
batch_size = 128
epochs = 15
input_shape = (32, 32, 3)

# the data, split between train and test sets
(X_train, Y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# x_train = np.empty([K_shots*num_classes ,28, 28],  dtype='uint8')
# y_train = np.empty([K_shots*num_classes],  dtype='uint8')
x_train = X_train[:K_shots*num_classes , :, :]
y_train = Y_train[:K_shots*num_classes]


for i in np.arange(10):
    class_ind = np.where(Y_train == i)
    class_ind = np.array(class_ind)
    class_ind.flatten()
    np.random.shuffle( class_ind )
    x_train[ i*K_shots:(i+1)*K_shots, :, :] = X_train[ class_ind[0,:K_shots], :, : ]
    y_train[ i*K_shots:(i+1)*K_shots ] = Y_train[ class_ind[0,:K_shots] ]



# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
# x_train = np.expand_dims(x_train, -1)
# x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

def prepare_data_for_resnet50(data):
    # data = data_to_transform.copy().values
    data = data.reshape(-1, 28, 28) / 255
    data = np.stack([data, data, data], axis=-1)
    return data

x_train = prepare_data_for_resnet50(x_train)
x_test = prepare_data_for_resnet50(x_test)

x_train = tf.keras.layers.ZeroPadding2D(padding=2)(x_train)
x_test = tf.keras.layers.ZeroPadding2D(padding=2)(x_test)

print(x_train.shape)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

base_model = tf.keras.applications.DenseNet169(include_top=False)
# base_model.summary()
inputs = keras.Input(shape=input_shape)
x = base_model(inputs, training=True)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.BatchNormalization()(x)
outputs = keras.layers.Dense(num_classes, activation="softmax", name="output_layer2")(x)
model = keras.Model(inputs, outputs)


model.summary()
# print(Myactivation)



# model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.compile(loss=Myloss, optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
