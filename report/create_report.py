import os
import sys
import subprocess

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

sys.path.append(os.path.abspath('.'))

from models.train import TRAINED_MODELS_PATH, DATA_PATH
from models.classifier import *
from models.utils import *
from preprocessing.main import process

# Constants defining file paths and commands
MAIN_TEX: str = os.path.join(os.getcwd(), 'report/main_template.tex')
OUTPUT_TEX: str = os.path.join(os.getcwd(), 'output/main.tex')
TEX_IMAGES_PATH: str = os.path.join(os.getcwd(), 'report/images')
BIRD_IMAGE_PATH: str = os.path.join(TEX_IMAGES_PATH, 'bird_test.jpg')
SAVE_CONF_MATRIX_PATH: str = os.path.join(TEX_IMAGES_PATH, 'output.jpg')
MODEL_PLOT_PATH: str = os.path.join(TEX_IMAGES_PATH, 'model_plot.jpg')
COMMAND: str = 'pdflatex -output-directory ' + os.path.join(os.getcwd(), 'output') + ' ' + OUTPUT_TEX

models = {
    'Random Forest': ClassifierRF,
    'SVM': ClassifierSVM,
    'CNN': ClassifierCNN
}


def load_file(path):
    """Loads and returns the contents of a file."""
    with open(path, 'r') as file:
        return "".join(file.readlines())


def generate_report(selected_model, img):
    """Generates a report based on the selected model and image.

    Args:
    - selected_model (str): The selected classification model.
    - img (Image): The input image for classification.

    Raises:
    - Exception: If unable to generate the report PDF.
    """
    # Construct the path to the selected model
    model_path = Path(os.path.join(TRAINED_MODELS_PATH, CLASSIFIERS[selected_model]))
    # Process the input image
    processed_img = process(img)

    model = models[selected_model]
    predicted_class, prob_scores = model.predict(model_path, img)
    print(predicted_class, prob_scores, sep='\n')

    # Generate classifier report and confusion matrix
    classifier_report = model.get_classifier_report(model_path)
    print(classifier_report)

    # if selected_model == 'CNN':
    #     handle_cnn(model_path, processed_img)
    #     return

    # Generate respective plots based on the selected model
    # if selected_model == 'Random Forest':
    #     generate_rf_plot(X_test, y_test, MODEL_PLOT_PATH)
    # elif selected_model == 'SVM':
    #     generate_svm_plot(X_test, y_test, MODEL_PLOT_PATH)
    # else:
    #     generate_cnn_plot(X_test, y_test, MODEL_PLOT_PATH)

    # Load the LaTeX main template file
    tex_code = load_file(MAIN_TEX)

    # Data dictionary to replace placeholders in the LaTeX template
    data = {
        'modelpredictedspecies': predicted_class[0],
        'classifiername': selected_model,
        'probscore1': prob_scores[0][0],
        'probscore2': prob_scores[0][1],
        'probscore3': prob_scores[0][2],
        'probscore4': prob_scores[0][3],
        'probscore5': prob_scores[0][4],
        'sklearnreport': classifier_report,
        'bird_test': BIRD_IMAGE_PATH,
        'conf_matrix': model_path / 'conf.jpg',
        'accu_plot': model_path / 'accuracy.png'
    }

    # Save the bird image
    img.save(BIRD_IMAGE_PATH)

    # Replace placeholders in the LaTeX template with actual values
    for key, value in data.items():
        tex_code = tex_code.replace(key, str(value))

    # Write the updated LaTeX template to the output file
    with open(OUTPUT_TEX, 'w') as f:
        f.write(tex_code)

    # Run the command to generate the report PDF and capture the output
    output = subprocess.run(COMMAND, shell=True, capture_output=True, text=True)

    # Print the output
    if f'Output written on {OUTPUT_TEX}'.replace('.tex', '.pdf') not in output.stdout:
        raise Exception('Unable to generate report pdf')


def generate_rf_plot(X_test, y_test, model_plot_path):
    """Generates a plot depicting Random Forest accuracy over the number of trees.

    Args:
    - X_test (np.ndarray): Test features.
    - y_test (np.ndarray): Test labels.
    - model_plot_path (str): Path to save the generated plot.
    """
    # List to store accuracy values
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

    # Save the generated plot
    fig = plt.gcf()
    plt.figure()
    fig.savefig(model_plot_path)


def generate_svm_plot(X_test, y_test, model_plot_path):
    """Generates a plot depicting SVM accuracy over different values of the regularization parameter (C).

    Args:
    - X_test (np.ndarray): Test features.
    - y_test (np.ndarray): Test labels.
    - model_plot_path (str): Path to save the generated plot.
    """
    # Different values of C to test
    C_values = [0.001, 0.01, 0.1, 1, 10, 100]
    accuracy_values = []

    # Iterating through different C values
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

    # Save the generated plot
    fig = plt.gcf()
    plt.figure()
    fig.savefig(model_plot_path)


def generate_cnn_plot(X_test, y_test, model_plot_path):
    """Generates a plot for CNN (Convolutional Neural Network) accuracy (to be implemented).

    Args:
    - X_test (np.ndarray): Test features.
    - y_test (np.ndarray): Test labels.
    - model_plot_path (str): Path to save the generated plot.
    """
    pass


img = Image.open('data/Little Egret/61001311.jpg')
generate_report('Random Forest', img)
