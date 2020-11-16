import keras as k
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from data_manipulator import load_dataset
import filter_funcs as ff
import os
from clr_callback import CyclicLR as CLR


train_dataset, test_dataset, validation_dataset = load_dataset() 
"""
    Model config:  
        - convolutional layer
            applies 32 3x3 filters to the input image
        - activation layer 

"""

# Used to maximise GPU utilisation. Model would not run on a GTX 1650
# with roughly 4Gb of VRAM without these config options
config = tf.ConfigProto()

config.gpu_options.allow_growth = True

session = tf.Session(config=config)

k.backend.tensorflow_backend.set_session(session)
# CNN attributes
batch_size = 200
num_classes = 10
epochs = 560

# dataset attributes
labels_index = 0
data_index = 1

# algorithms used
# used to perform identification of non-linear data
rectifier_function = "relu"

# used to perform classifications of non-linear data
softmax_function = "softmax"

train_classes=train_dataset[0]
train_data=train_dataset[1]

test_classes=test_dataset[0]
test_data=test_dataset[1]

validation_classes=validation_dataset[0]
validation_data=validation_dataset[1]

# pre-process images to help avoid overfitting or underfitting

# used to reduce overall variance of dataset values
train_data = ff.RGB_to_greyscale(train_data)
validation_data = ff.RGB_to_greyscale(validation_data)
test_data = ff.RGB_to_greyscale(test_data)

#model building

train_classes = k.utils.to_categorical\
(
    train_classes,
    num_classes
)

validation_classes = k.utils.to_categorical\
(
    validation_classes,
    num_classes
)

test_classes = k.utils.to_categorical\
(
    test_classes,
    num_classes
)

CNN = k.Sequential()

dataset_shape = (test_data.shape[1:]) #/image resolution

img_filters = 32
filter_shape = (3, 3) 

CNN.add(k.layers.Conv2D
    (
        img_filters, filter_shape, padding = "same", input_shape=(dataset_shape))
    )

# Activation function == calculating a weighted sum of input nodes, bias added,
# and decides whether input nodes should be 'fired as output' to the next layer
CNN.add(k.layers.Activation(rectifier_function))

# perform max pooling on generated features maps, reduces complexity
# of the feature map matrices.
CNN.add(k.layers.AveragePooling2D(pool_size=(2,2)))

img_filters = 64

CNN.add(k.layers.Conv2D(img_filters, filter_shape, padding="same"))

CNN.add(k.layers.Activation(rectifier_function))

CNN.add(k.layers.AveragePooling2D(pool_size=(2,2)))

CNN.add(k.layers.Dropout(0.2))

img_filters = 128

CNN.add(k.layers.Conv2D(img_filters, filter_shape, padding="same"))

CNN.add(k.layers.Activation(rectifier_function))

CNN.add(k.layers.AveragePooling2D(pool_size=(2,2)))

CNN.add(k.layers.Dropout(0.2))

img_filters = 256

CNN.add(k.layers.Conv2D(img_filters, filter_shape, padding="same"))

CNN.add(k.layers.Activation(rectifier_function))

CNN.add(k.layers.AveragePooling2D(pool_size=(2,2)))

CNN.add(k.layers.Dropout(0.7))

img_filters = 512

CNN.add(k.layers.Conv2D(img_filters, filter_shape, padding="same"))
CNN.add(k.layers.Conv2D(img_filters, filter_shape, padding="same"))


CNN.add(k.layers.Activation(rectifier_function))

CNN.add(k.layers.MaxPooling2D(pool_size=(2,2)))

# Flatten the pooled images into a continuous vector.
# i.e. Taking the pooled matrix and converting to 1-D vector.
CNN.add(k.layers.Flatten())

# Connect the nodes retrieved from flattening the pooled matrix.
# Nodes will be used as an input layer for the fully-connected layers.
# Note to self: change through model experimentation!
num_nodes = 512

CNN.add(k.layers.Dense(units = num_nodes, activation = rectifier_function))

# initalise the output/classification layer

CNN.add(k.layers.Dense(units = num_classes, activation = softmax_function))

# Optimization algorithm == Given function f(x), optimization algorithm used to 
# minimize or maximise the value of f(x). Optimization algorithms used to optimize cost 
# functions - learning of function f so f(X) maps to/predicts y.
optimizer = k.optimizers.Adam()

CNN.compile(optimizer = optimizer, loss = "categorical_crossentropy",\
    metrics=["accuracy"])

### model fitting ###

# create new data from the images provided by performing operations,
# e.g. flipping the image.

data_preprocess = ImageDataGenerator(
    horizontal_flip=True,
    zoom_range=[1.0, 1.2],
    )
data_preprocess.fit(train_data)
# data generator used to handle the processing of large dataset across
# multiple system cores. 
test_img_instances=40000
validation_img_instances=10000

# Epochs == the amount of times a dataset is passed forward and
# backward through the network.

# Batch == The set/parts the dataset is split into.
# Defines the amount of images passed into the CNN at once.

# Iterations == The amount of batches needed to complete an epoch.
# Batch number == iterations for one epoch 

iterations_in_epoch = test_img_instances / batch_size

try:
    step_size = 8 * iterations_in_epoch
    clr = CLR(step_size=step_size, base_lr=0.00001, max_lr=0.0012)

    try:
        CNN.fit_generator\
            (
                data_preprocess.flow(train_data, train_classes,
                batch_size=batch_size), callbacks=[clr],
                epochs=epochs,
                steps_per_epoch=iterations_in_epoch,
                validation_data=(validation_data, validation_classes)
            )
    except Exception as e:
        print("Error when fitting model:")
        print(e)

    # lower loss == better model.
    try:
        model_name = input("Enter model name for model save:")
        model_file_format = ".h5"
        CNN.save("{}/models/{}{}".format(os.getcwd(), model_name, model_file_format))
    except Exception as e:
        print("Error when saving model:")
        print(e)

    scores = CNN.evaluate(test_data, test_classes, batch_size=batch_size)
    print("Test loss:", scores[0])
    print("Test accuracy:", scores[1])
finally:
    k.backend.clear_session()

# TODO: Gen fit graph.