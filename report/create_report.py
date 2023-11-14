import os
import subprocess

from models.classifier import *
# from models.classifier import TRAINED_MODELS_PATH
# from models.utils import load_model
from preprocessing.main import process
from models.utils import *
# from models.utils import get_classifier_report
# from models.utils import load_flattened_data


MAIN_TEX = os.path.join(os.getcwd(), 'report/main_template.tex')
OUTPUT_TEX = os.path.join(os.getcwd(), 'output/main.tex')
TEX_IMAGES_PATH = os.path.join(os.getcwd(), 'report/images')
BIRD_IMAGE_PATH = os.path.join(TEX_IMAGES_PATH, 'bird_test.jpg')
SAVE_CONF_MATRIX_PATH = os.path.join(TEX_IMAGES_PATH, 'output.jpg')
MODEL_PLOT_PATH = os.path.join(TEX_IMAGES_PATH, 'model_plot.jpg')
COMMAND = 'pdflatex -output-directory ' + os.path.join(os.getcwd(), 'output') + ' ' + OUTPUT_TEX 

def load_file(path):
    with open(path, 'r') as file:
        return "".join(file.readlines())

def generate_report(selected_model, img):
    model_path = os.path.join(TRAINED_MODELS_PATH, CLASSIFIERS[selected_model])

    classifier = load_model(model_path)

    processed_img = process(img)

    if selected_model in ['Random Forest', 'SVM']:
        classifier_input = flatten_image(processed_img)
        classifier_input = classifier_input.reshape(1, classifier_input.shape[0])

    else:
        # handle image process for cnn model
        pass

    X_test, y_test = load_flattened_data(DATA_PATH, SPECIES, split = 'test')
    
    if selected_model == 'Random Forest':
        generate_rf_plot(X_test, y_test, MODEL_PLOT_PATH)
    elif selected_model == 'SVM':
        generate_svm_plot(X_test, y_test, MODEL_PLOT_PATH)
    else:
        generate_cnn_plot(X_test, y_test, MODEL_PLOT_PATH)

    predicted_class = classifier.predict(classifier_input)

    prob_scores = classifier.predict_proba(classifier_input)
    
    classifier_report = get_classifier_report(X_test, y_test, SPECIES, model_path, SAVE_CONF_MATRIX_PATH)
    
    tex_code = load_file(MAIN_TEX)

    data = {
        'modelpredictedspecies' : predicted_class[0],
        'classifiername' : selected_model,
        'probscore1' : prob_scores[0][0],
        'probscore2' : prob_scores[0][1],
        'probscore3' : prob_scores[0][2],
        'probscore4' : prob_scores[0][3],
        'probscore5' : prob_scores[0][4],
        'sklearnreport' : classifier_report,
        'bird_test' : BIRD_IMAGE_PATH,
        'conf_matrix' : SAVE_CONF_MATRIX_PATH,
        'accu_plot' : MODEL_PLOT_PATH
    }
    
    img.save(BIRD_IMAGE_PATH)

    for key, value in data.items():
        tex_code = tex_code.replace(key, str(value))

    with open(OUTPUT_TEX, 'w') as f:
        f.write(tex_code)

    # Run the command and capture output
    output = subprocess.run(COMMAND, shell=True, capture_output=True, text=True)

    # Print the output
    if f'Output written on {OUTPUT_TEX}'.replace('.tex', '.pdf') not in output.stdout:
        raise Exception('Unable to generate report pdf')
    
def generate_rf_plot(X_test, y_test, model_plot_path):

    accuracy_values = []

    # Training Random Forest with different numbers of trees
    for n_trees in range(1, 101):
        rf_model = RandomForestClassifier(n_estimators=n_trees, random_state=42)
        rf_model.fit(X_test, y_test)
        accuracy = rf_model.score(X_test, y_test)
        accuracy_values.append(accuracy)

    # Plotting accuracy over the number of trees
    plt.plot(range(1, 101), accuracy_values, marker='o')
    plt.title('Random Forest Accuracy over Number of Trees')
    plt.xlabel('Number of Trees')
    plt.ylabel('Accuracy')
    
    fig = plt.gcf()
    plt.figure()
    fig.savefig(model_plot_path)

def generate_svm_plot(X_test, y_test, model_plot_path):
    C_values = [0.001, 0.01, 0.1, 1, 10, 100]  # Different values of C to test
    accuracy_values = []

    for C in C_values:
        svm_model = SVC(kernel='rbf', C=C, random_state=42)
        svm_model.fit(X_test, y_test)
        accuracy = svm_model.score(X_test, y_test)
        accuracy_values.append(accuracy)

    # Plotting accuracy over different C values
    plt.plot(C_values, accuracy_values, marker='o')
    plt.xscale('log')  # Log scale for better visualization of C values
    plt.title('SVM Accuracy over Different C Values (Linear Kernel)')
    plt.xlabel('C (Regularization Parameter)')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    fig = plt.gcf()
    plt.figure()
    fig.savefig(model_plot_path)


def generate_cnn_plot(X_test, y_test, model_plot_path):
    pass

img = Image.open('/home/sarvam/Documents/cs_699/CS_699/data/Blue Jay/47351251.jpg')
generate_report('SVM', img)






    



