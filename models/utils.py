import numpy as np
from PIL import Image
import os
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from mlxtend.plotting import heatmap
from pathlib import Path


def flatten_image(image: Image) -> np.ndarray:
    """
    Flatten the input grayscale image and normalize pixel values to [0, 1].

    Parameters:
    - image (Image): Grayscale image.

    Returns:
    - np.ndarray: Flattened and normalized pixel values.
    """
    # Read the image as grayscale
    image_greyscale = image.convert('L')

    # Check if the image is successfully loaded
    if image_greyscale is not None:
        # Normalize the image to the range [0, 1]
        normalized_image = np.array(image_greyscale) / 255.0

        # Now 'normalized_image' contains the pixel values normalized to [0, 1] and returned as flattened array
        return normalized_image.flatten()
    else:
        raise Exception("Image is None. Expected a valid Image object.")


def load_flattened_data(folder_path: str, species: list, split: str = 'train') -> tuple:
    """
    Load flattened images and their corresponding labels from the specified folder.

    Parameters:
    - folder_path (str): Path to the main folder containing subfolders for each species.
    - species (list): List of species names.
    - split (str): Dataset split ('train' or 'test'). Default is 'train'.

    Returns:
    - tuple: Tuple containing numpy arrays of images and labels.
    """
    images = []
    labels = []

    for type in species:
        for filename in os.listdir(os.path.join(folder_path, split, type)):
            if filename.endswith(".jpg"):
                img = Image.open(os.path.join(folder_path, split, type, filename))
                images.append(flatten_image(img))  # Flatten and append the processed image
                labels.append(type)  # Append the corresponding label

    return np.array(images), np.array(labels)


def get_classifier_report(X_test: np.ndarray, y_test: np.ndarray, species: list, model_path: str,
                          save_conf_matrix_path: str) -> str:
    """
    Generate a classification report and save the confusion matrix plot.

    Parameters:
    - X_test (np.ndarray): Test data.
    - y_test (np.ndarray): True labels for the test data.
    - species (list): List of species names.
    - model_path (str): Path to the trained model file.
    - save_conf_matrix_path (str): Path to save the confusion matrix plot.

    Returns:
    - str: Classification report.
    """
    classifier = load_model(model_path)  # Load the trained model
    y_pred = classifier.predict(X_test)  # Predict using the test data
    report = classification_report(y_test, y_pred)  # Generate the classification report

    # Generate and save the confusion matrix plot
    conf_mat = confusion_matrix(y_test, y_pred)
    heatmap(conf_mat, column_names=species, row_names=species, figsize=(20, 20), cmap="BuPu")

    fig = plt.gcf()  # Get the current figure
    plt.figure()
    fig.savefig(save_conf_matrix_path)  # Save the confusion matrix plot

    return report  # Return the classification report


def load_model(model_path: str):
    """
    Load a trained classifier model from the specified file.

    Parameters:
    - model_path (str): Path to the trained model file.

    Returns:
    - object: Trained classifier model.
    """
    with open(Path(model_path) / 'model.pkl', 'rb') as f:
        classifier = pickle.load(f)

        return classifier


def save_model(model_path: str, classifier):
    """
    Save a trained classifier model to the specified file.

    Parameters:
    - model_path (str): Path to save the trained model file.
    - classifier: Trained classifier model.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        model_path.mkdir(parents=True)
    with open(Path(model_path) / 'model.pkl', 'wb') as f:
        pickle.dump(classifier, f)
