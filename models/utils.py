import numpy as np
from PIL import Image
import os
import pickle
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from mlxtend.plotting import heatmap

def flatten_image(image: Image) -> np.array:

    # Read the image as grayscale
    
    image_greyscale = image.convert('L')

    # Check if the image is successfully loaded
    if image_greyscale is not None:
        # Normalize the image to the range [0, 1]
        normalized_image = np.array(image_greyscale) / 255.0

        # Now 'normalized_image' contains the pixel values normalized to [0, 1] and returned as flattened array
        return normalized_image.flatten()
    else:
        raise Exception(f"image is None, expected valid Image object")

def load_flattened_data(folder_path, species, split = 'train'):
    images = []
    labels = []

    for type in species:
        for filename in os.listdir(os.path.join(folder_path, type, split)):
            if filename.endswith(".jpg"):  
                img = Image.open(os.path.join(folder_path, type, split,  filename))
                images.append(flatten_image(img))
                labels.append(type) 

    return np.array(images), np.array(labels)


def get_classifier_report(X_test, y_test, species, model_path, save_conf_matrix_path):

    with open(model_path, 'rb') as f:
        classifier = pickle.load(f)
    
    y_pred = classifier.predict(X_test)

    report = classification_report(y_test, y_pred)

    conf_mat = confusion_matrix(y_test, y_pred)
    heatmap(conf_mat, column_names=species, row_names=species, figsize=(20,20), cmap="BuPu")
    plt.savefig(save_conf_matrix_path)

    return report

def load_model(model_path):
    with open(model_path, 'rb') as f:
        classifier = pickle.load(f)

def save_model(model_path, classifier):
    with open(model_path, 'wb') as f:
        pickle.dump(classifier, f)
