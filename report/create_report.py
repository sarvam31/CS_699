import os

from models.classifier import *
# from models.classifier import TRAINED_MODELS_PATH
# from models.utils import load_model
from preprocessing.main import process
from models.utils import *
# from models.utils import get_classifier_report
# from models.utils import load_flattened_data

MAIN_TEX = './report/main.tex'
IMAGES_PATH = './report/images/'

def load_file(path):
    with open(path, 'r') as file:
        return file.readlines()

def generate_report(selected_model, img):
    classifier = load_model(os.path.join(TRAINED_MODELS_PATH, CLASSIFIERS[selected_model]))

    processed_img = process(img)

    if selected_model in ['Random Forest', 'SVM']:
        classifier_input = flatten_image(processed_img)
    
    else:
        # handle image process for cnn model
        pass

    predicted_class = classifier.predict(classifier_input)

    prob_scores = classifier.predict_proba(classifier_input)

    X_test, y_test = load_flattened_data(DATA_PATH, SPECIES, split = 'test')

    rf_model = os.path.join(TRAINED_MODELS_PATH, CLASSIFIERS['Random Forest'])
    classifier_report = get_classifier_report(X_test, y_test, SPECIES, rf_model, SAVE_CONF_MATRIX_PATH)

    tex_code = load_file(MAIN_TEX)

    print(tex_code)






    



