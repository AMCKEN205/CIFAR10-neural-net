from keras.datasets import cifar10
from numpy import ndarray as numpy_ar
from PIL import Image as img_gen
from sklearn.model_selection import train_test_split as train_validation_split

def load_dataset():
    """ loads the cifar 10 image dataset.
    Loads 50000 training images and 10000 test images.
    Returns the data as ndarrays.

    In this use case, a validation dataset is created using 20%
    of the training images. This gives a final split of 40000 training
    10000 validation and 10000 testing. (66.7/16.7/16.7)
    
    The shape of the numpy array data dataset returned is:
        num of entries (e.g. 50000 for training)
        32 (height),
        32 (width),
        3 (colour channels (RGB))
    
    The shape of the numpy array labels dataset returned is:
        num of entries (e.g. 50000 for training)
        1 (label/category/class)

    Dataset uses:

    Training: Will be used to fit the parameters of the classifier.

    Validation: Will be used to fine tune the parameters of the classifier.

    Test: Will be used to assess the performance of the final model.

    """

    (data_train_set, labels_train_set), \
        (data_test_set, labels_test_set) = cifar10.load_data()
    
    # get the validation dataset
    t_v_data = data_train_set
    t_v_labels = labels_train_set
    data_train_set = None
    labels_train_set = None

    validation_size = 0.20 # 20% of the training data

    # split set
    data_train_set, data_validation_set = train_validation_split\
        (
            t_v_data, test_size=validation_size, shuffle=False
        )

    labels_train_set, labels_validation_set = train_validation_split\
        (
            t_v_labels, test_size=validation_size, shuffle=False
        )
        e colour channel, so add it back in.
        # greyscale images to reduce vari
    validation_dataset = (labels_validation_set, data_validation_set)

    return train_dataset, test_dataset, validation_dataset
    

class Image:
    class_label : int
    pixel_matrix : numpy_ar

    def __init__(self, class_label : int, pixel_matrix : numpy_ar, labels_ar : numpy_ar):
        self.class_label = class_label
        self.pixel_matrix = pixel_matrix
        self.labels_ar = labels_ar

    def get_image(self):
        colour_encoding = "RGB"
        img = img_gen.fromarray(self.pixel_matrix, colour_encoding)
        return img


load_dataset()