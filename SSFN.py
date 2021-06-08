import numpy as np
from tensorflow import keras
import tensorflow as tf
from MyFunctions import *

Myactivation='softmax'
Myloss='categorical_crossentropy'

parameters_path = "./parameters/"

SSFN_hparameters = {}
SSFN_hparameters["data"] = "MNIST"
SSFN_hparameters["lam"] = 10**(2)
SSFN_hparameters["mu"] = 10**(3)
SSFN_hparameters["kMax"] = 100
SSFN_hparameters["NodeNum"] = 1000
SSFN_hparameters["LayerNum"] = 1

# Model / data parameters
niter = 10
p = 0.01
K_shots = 100
num_classes = 10
batch_size = 128
epochs = 15

test_accuracy = np.empty([1, niter])

data = 'mnist'

if data=='mnist':
    input_shape = (28, 28, 1)
elif data=='cifar10':
    input_shape = (32, 32, 3)


for iter in np.arange(niter):
    tf.keras.backend.clear_session()
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
    # print("x_train shape:", x_train.shape)
    # print(x_train.shape[0], "train samples")
    # print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes).T
    y_test = keras.utils.to_categorical(y_test, num_classes).T

    x_train = np.reshape(x_train, (K_shots*num_classes, 784)).T
    x_test = np.reshape(x_test, (10000, 784)).T
    
    
    train_acc, test_acc = SSFN_train( x_train, x_test, y_train, y_test, SSFN_hparameters)
    test_accuracy[0, iter] = test_acc
    print(test_accuracy[0,:iter+1])

print(np.mean(test_accuracy))

