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
# X_test, y_test = utils.load_flattened_data(DATA_PATH, SPECIES, split = 'test')

TRAINED_MODELS_PATH = os.path.join(os.getcwd(), 'models/trained_models')

def train_RF_classifier(X_train: np.ndarray, y_train: np.ndarray, save_model_path: str):
    """
    Train a Random Forest classifier.

    Parameters:
    - X_train (np.ndarray): Training data.
    - y_train (np.ndarray): Training labels.
    - save_model_path (str): Path to save the trained model file.
    """
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)
    save_model(os.path.join(save_model_path, CLASSIFIERS['Random Forest']), classifier)

    print('Random forest classifier training complete')


def train_SVM_classifier(X_train: np.ndarray, y_train: np.ndarray, save_model_path: str):
    """
    Train an SVM classifier.

    Parameters:
    - X_train (np.ndarray): Training data.
    - y_train (np.ndarray): Training labels.
    - save_model_path (str): Path to save the trained model file.
    """
    classifier = SVC(probability=True)
    classifier.fit(X_train, y_train)
    save_model(os.path.join(save_model_path, CLASSIFIERS['SVM']), classifier)

    print('SVM classifier training complete')
    

# def train_CNN_classifier(X_train, y_train, save_model_path):

#     classifier = CNN_classifier()

#     with open(os.path.join(save_model_path, "cnnclassifier.pkl"), 'wb') as f:
#         pickle.dump(classifier, f)







