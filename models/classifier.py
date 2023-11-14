from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import models.utils as utils
import os
from models.utils import save_model
import numpy as np

SPECIES = ['Blue Jay', 'Grey Heron', 'Indian Peafowl', 'Little Egret', 'Red-Vented Bulbul']

CLASSIFIERS = {
    'Random Forest' : 'randomforestclassifier.pkl',
    'SVM' : 'svmclassifier.pkl',
    'CNN' : 'cnnclassifier.pkl' }


DATA_PATH = os.path.join(os.getcwd(), 'processed') 

X_train, y_train = utils.load_flattened_data(DATA_PATH, SPECIES, split = 'train')

TRAINED_MODELS_PATH = os.path.join(os.getcwd(), 'models/trained_models')

def train_RF_classifier(X_train: np.ndarray, y_train: np.ndarray, save_model_path: str):
    """
    Train a Random Forest classifier.

    Parameters:
    - X_train (np.ndarray): Training data.
    - y_train (np.ndarray): Training labels.
    - save_model_path (str): Path to save the trained model file.
    """
    classifier = RandomForestClassifier()  # Initialize Random Forest classifier
    classifier.fit(X_train, y_train)  # Train the classifier
    save_model(os.path.join(save_model_path, CLASSIFIERS['Random Forest']), classifier)  # Save the trained model

    print('Random forest classifier training complete')  # Display completion message


def train_SVM_classifier(X_train: np.ndarray, y_train: np.ndarray, save_model_path: str):
    """
    Train an SVM classifier.

    Parameters:
    - X_train (np.ndarray): Training data.
    - y_train (np.ndarray): Training labels.
    - save_model_path (str): Path to save the trained model file.
    """
    classifier = SVC(probability=True)  # Initialize SVM classifier
    classifier.fit(X_train, y_train)  # Train the classifier
    save_model(os.path.join(save_model_path, CLASSIFIERS['SVM']), classifier)  # Save the trained model

    print('SVM classifier training complete')  # Display completion message

    

# def train_CNN_classifier(X_train, y_train, save_model_path):

#     classifier = CNN_classifier()

#     with open(os.path.join(save_model_path, "cnnclassifier.pkl"), 'wb') as f:
#         pickle.dump(classifier, f)







