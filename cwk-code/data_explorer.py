import data_manipulator as dm
import re as regex
from collections import Counter
import numpy as np
from pandas import DataFrame
import filter_funcs as ff

train_dataset, test_dataset, validation_dataset = dm.load_dataset()

class_labels = {
        "0" : "airplane",
        "1" : "automobile",
        "2" : "bird",
        "3" : "cat",
        "4" : "deer",
        "5" : "dog",
        "6" : "frog",
        "7" : "horse",
        "8" : "ship",
        "9" : "truck"
}

def get_inter_class_variance(dataset : tuple, filter_funcs : list = None):
        """ Gets the variance between classes.
        Used to analyse difference between image classes with filters applied.
        Ideally want to maximise inter class variance, as this
        makes images of different classes more distinct from one another.
        Much easier to differentiate! """
        
        dataset_filtered = dataset
        if filter_funcs is not None:
                for filter_func in filter_funcs:
                        dataset_filtered = filter_func(dataset[1])
        else:
                dataset_filtered = dataset[1]
                
        return np.var(dataset_filtered)

def get_intra_class_variance(dataset : tuple, filter_funcs: list = None):
        """ Gets the variance within individual classes.
        Used to analyse diference within each image class.
        Ideally, this value should be minimised, as this makes it easier to
        identify that patterns that group together images of each class."""

        dataset_filtered = dataset
        if filter_funcs is not None:
                for filter_func in filter_funcs:
                        dataset_filtered = (dataset[0], filter_func(dataset[1]))

        for _class, label in class_labels:
                for 

        print("for debug")

        
def get_dataset_variance_stats():
        #TODO
        print("TODO")

def get_dataset_stats():
        """ Get the distribution of data across the dataset """

        print("\nTraining instances = 40000 - 66.7%\n")
        print ("Test instances = 10000 - 16.7%\n")
        print("Validation instances = 10000 - 16.7%\n")

        datasets = {
                "Training set" : train_dataset,
                "Validation set" : validation_dataset,
                "Test set" : test_dataset
        }

        dataset_instances = {
                "Training set" : 40000,
                "Validation set" : 10000,
                "Test set" : 10000
        }

        for dataset_name, dataset in datasets.items():
                print("\nStats for {}".format(dataset_name))
                class_nums, counts = np.unique(dataset[0], return_counts=True)
                class_nums = list(map(str, class_nums))
                counts = list(map(int, counts))
                percent_of_dataset = list()
                class_names = list()

                for count in counts:
                        percent_of_dataset.append\
                        (
                                count/dataset_instances[dataset_name]
                        )

                for class_num in class_nums:
                        class_names.append(class_labels[class_num])

                stats = \
                {
                        "class" : class_names,
                        "occurances" : counts,
                        "percent of dataset" : percent_of_dataset
                }

                df = DataFrame(stats)
                print(df)
                df.to_csv("{}_stats.csv".format(dataset_name))

get_intra_class_variance(train_dataset)
print(str("{} normalised train images variance".format(get_inter_class_variance(train_dataset, [ff.normalize_pixel_values]))))