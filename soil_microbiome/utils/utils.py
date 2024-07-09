import os
import numpy as np
import pandas as pd
from soil_microbiome import global_vars
import matplotlib.pyplot as plt
from functools import wraps
import time
import random
from geopy.distance import great_circle


def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created successfully.")

    else:
        print(f"Directory '{directory}' already exists.")


def check_extension(filename, extension):
    # Convert the filename to lowercase to make the comparison case-insensitive
    lowercase_filename = filename.lower()

    # Check if the filename already has the extension
    if not lowercase_filename.endswith(extension):
        # Add the extension to the filename
        filename += extension

    file = os.path.join(global_vars['data_dir'], filename)

    return file


def read_file(filename):
    name, ext = os.path.splitext(filename)

    if ext not in ['.xlsx', '.csv']:
        raise ValueError('Please specify either xlsx or csv file.')

    full_path = os.path.join(global_vars['data_dir'], filename)

    if not os.path.isfile(full_path):
        raise ValueError('File does not exist.')

    if ext == '.xlsx':
        data = pd.read_excel(full_path)

    else:
        data = pd.read_csv(full_path)

    return data


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'{func.__name__} : {total_time:.4f} seconds')
        return result

    return timeit_wrapper


def cross_validation(data, labels, k=5):
    fold_size = len(data) // k
    indices = list(range(len(data)))

    # Shuffle indices to ensure randomness
    import random
    random.shuffle(indices)

    for i in range(0, len(data), fold_size):
        test_indices = indices[i:i + fold_size]
        train_indices = indices[:i] + indices[i + fold_size:]

        test_data = [data[j] for j in test_indices]
        test_labels = [labels[j] for j in test_indices]

        train_data = [data[j] for j in train_indices]
        train_labels = [labels[j] for j in train_indices]

        # Train your model with train_data and train_labels here

        # Test your model with test_data and compare predictions with test_labels here
        # You can calculate performance metrics like accuracy, precision, recall, etc.

        print(f"Train on {len(train_data)} samples, Test on {len(test_data)} samples")


def haversine(coord1, coord2):
    return great_circle(coord1, coord2).km
