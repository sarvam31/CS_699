from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import utils
import os
import pickle
from utils import save_model

SPECIES = ['Blue Jay', 'Grey Heron', 'Indian Peafowl', 'Little Egret', 'Red-Vented Bulbul']

CLASSIFIERS = {
    'Random Forest' : 'randomforestclassifier.pkl',
    'SVM' : 'svmclassifier.pkl',
    'CNN' : 'cnnclassifier.pkl' }

DATA_PATH = './CS_699/processed/'

X_train, y_train = utils.load_flattened_data(DATA_PATH, SPECIES, split = 'train')
# X_test, y_test = utils.load_flattened_data(DATA_PATH, SPECIES, split = 'test')

TRAINED_MODELS_PATH = './CS_699/models/trained_models'

SAVE_CONF_MATRIX_PATH = './CS_699/report/output.jpg'

def train_RF_classifier(X_train, y_train, save_model_path):
    classifier = RandomForestClassifier()

    classifier.fit(X_train, y_train)

    save_model(os.path.join(save_model_path, CLASSIFIERS['Random Forest']), classifier)

def train_SVM_classifier(X_train, y_train, save_model_path):
    classifier = SVC()

    classifier.fit(X_train, y_train)

    save_model(os.path.join(save_model_path, CLASSIFIERS['SVM']), classifier)
    

# def train_CNN_classifier(X_train, y_train, save_model_path):

#     classifier = CNN_classifier()

#     with open(os.path.join(save_model_path, "cnnclassifier.pkl"), 'wb') as f:
#         pickle.dump(classifier, f)




train_RF_classifier(X_train, y_train, TRAINED_MODELS_PATH)
train_SVM_classifier(X_train, y_train, TRAINED_MODELS_PATH)
# train_CNN_classifier(X_train, y_train, trained_models_path)

# rf_model = os.path.join(TRAINED_MODELS_PATH, CLASSIFIERS['Random Forest'])
# svm_model = os.path.join(TRAINED_MODELS_PATH, CLASSIFIERS['SVM'])
# cnn_model = os.path.join(TRAINED_MODELS_PATH, CLASSIFIERS['CNN'])

# print(utils.get_classifier_report(X_test, y_test, SPECIES, rf_model, './CS_699/report/output.jpg'))
# print(utils.get_classifier_report(X_test, y_test, SPECIES, svm_model, './CS_699/report/output2.jpg'))
# print(utils.get_classifier_report(X_test, y_test, SPECIES, cnn_model, './CS_699/report/output3.jpg'))




