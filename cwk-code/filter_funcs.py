import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu as otsu

greyscaled_colour_channels = 1

def normalize_pixel_values(dataset_to_normalise : np.array):
    """ Pixel values can be fed into a neural network in original 
    format, however this can present model creation difficulties,
    such as slower model training. So normalise the values!"""
    
    type_to_use = "float32" # for normalised pixels

    dataset_normalised = list()

    for img in dataset_to_normalise:
        RGB_value_range = 255.0

        # Convesion from int to float, to allow for values
        # to exist within a range of 0 to 1 (z-score normalization)

        pixels_normalised = img.astype(type_to_use)

        # data normalisation from 0 to 1
        pixels_normalised /=  RGB_value_range

        dataset_normalised.append(pixels_normalised)

    return np.asarray(dataset_normalised)


def RGB_to_greyscale(dataset_to_greyscale : np.array):
    # Dot product greyscale code sourced from: 
    # https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
    
    img_width = 32
    img_height = 32

    dataset_shape_entries_pos = 0
    img_entries = dataset_to_greyscale.shape[dataset_shape_entries_pos]
    
    greyscale_img_shape = (img_width, img_height, greyscaled_colour_channels)
    greyscale_dataset_shape = (img_entries, img_width, img_height, greyscaled_colour_channels)

    dataset_greyscaled = np.empty(greyscale_dataset_shape)
    cur_img = 0
    for data in dataset_to_greyscale:
        greyscaled_img_nparray = data.dot([0.07, 0.72, 0.21])
        greyscaled_img_nparray = greyscaled_img_nparray.reshape(greyscale_img_shape)
        # Greyscaling seems to remove the colour channel, so add it back in.
        # greyscale images to reduce variance 
        # introduced by pixel colour values.
        dataset_greyscaled[cur_img] = greyscaled_img_nparray
        cur_img += 1

    return dataset_greyscaled

def resize_dataset(dataset_to_resize: np.array, new_size_shape: tuple):
    
    dataset_shape_entries_pos = 0
    img_entries = dataset_to_resize.shape[dataset_shape_entries_pos]
    greyscale_control_char = "L"
    dataset_resized_shape = (img_entries, new_size_shape[0], new_size_shape[1],\
        greyscaled_colour_channels)

    dataset_resized = np.empty(dataset_resized_shape)
    cur_img = 0
    for data in dataset_to_resize:
        # fromarray gives a too many dimensions error with
        # image colour channels included.
        data = data.reshape(data.shape[0], data.shape[1])
        image = Image.fromarray(data, greyscale_control_char)
        image = image.resize(new_size_shape)
        img_ar = np.array(image.getdata(), dtype=np.float32)

        # resize to ensure image array shape correct.
        img_ar = img_ar.reshape((new_size_shape[0], \
            new_size_shape[1], greyscaled_colour_channels))

        dataset_resized[cur_img] = img_ar

        cur_img += 1

    return dataset_resized

def otsu_filter(dataset_to_filter : np.array):
    dataset_shape_entries_pos = 0
    img_entries = dataset_to_filter.shape[dataset_shape_entries_pos]
    img_height = 32
    img_width = 32
    dataset_filtered_shape = (img_entries, img_height, img_width,\
    greyscaled_colour_channels)
    dataset_filtered = np.empty(dataset_filtered_shape)
    cur_img = 0
    for data in dataset_to_filter:
         # fromarray gives a too many dimensions error with
        # image colour channels included.
        otsu_mask = otsu(data)
        image_filtered = data < otsu_mask
        image_filtered = image_filtered.astype(np.float32)
        # resize to ensure image array shape correct.
        image_filtered = image_filtered.reshape((img_height, \
            img_width, greyscaled_colour_channels))

        dataset_filtered[cur_img] = image_filtered

        cur_img += 1

    return dataset_filtered
