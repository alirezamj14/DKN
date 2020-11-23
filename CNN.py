import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, UpSampling2D, Flatten, BatchNormalization, Dense, Dropout, GlobalAveragePooling2D

Myactivation='softmax'
Myloss='categorical_crossentropy'

parameters_path = "./parameters/"


# Model / data parameters
niter = 1
p = 0.01
K_shots = 50
num_classes = 10
batch_size = 128
epochs = 15

test_accuracy = np.empty([1, niter])

data = 'mnist'

if data=='mnist':
    input_shape = (28, 28, 1)
elif data=='cifar10':
    input_shape = (32, 32, 3)


inputs = keras.Input(shape=input_shape)
h = layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(inputs)
h = layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(h)
h = layers.MaxPooling2D(pool_size=(2, 2))(h)
h = layers.Dropout(0.25)(h)
h = layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(h)
h = layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(h)
h = layers.MaxPooling2D(pool_size=(2, 2))(h)
h = layers.Dropout(0.25)(h)
h = layers.Flatten()(h)
h = layers.Dense(512, activation="relu")(h)
h = layers.Dropout(0.5)(h)
outputs = layers.Dense(num_classes, activation=Myactivation, name="output_layer")(h)
model = Model(inputs=inputs, outputs=outputs)
model.summary()


# the data, split between train and test sets

# x_train = np.empty([K_shots*num_classes ,28, 28],  dtype='uint8')
# y_train = np.empty([K_shots*num_classes],  dtype='uint8')


for iter in np.arange(niter):
    tf.keras.backend.clear_session()
    model = Model(inputs=inputs, outputs=outputs)
    if data=='mnist':
        (X_train, Y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    elif data=='cifar10':
        (X_train, Y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()


    x_train = X_train[:K_shots*num_classes]
    y_train = Y_train[:K_shots*num_classes]

    for i in np.arange(10):
        class_ind = np.where(Y_train == i)
        class_ind = np.array(class_ind)
        class_ind.flatten()
        np.random.shuffle( class_ind )
        x_train[ i*K_shots:(i+1)*K_shots ] = X_train[ class_ind[0,:K_shots] ]
        y_train[ i*K_shots:(i+1)*K_shots ] = Y_train[ class_ind[0,:K_shots] ]


    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    # x_train = np.expand_dims(x_train, -1)
    # x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    # print(x_train.shape[0], "train samples")
    # print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)


    # print(Myactivation)

    # model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.compile(loss=Myloss, optimizer="adam", metrics=["accuracy"])

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0, validation_split=0.1)

    score = model.evaluate(x_test, y_test, verbose=0)
    # print("Test loss:", score[0])
    # print("Test accuracy:", score[1])

    test_accuracy[0,iter] = np.array(score[1])
    print(test_accuracy[0,:iter])

print(np.mean(test_accuracy))
