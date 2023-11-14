import os
import subprocess

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from models.classifier import *
from preprocessing.main import process
from models.utils import *


# Constants defining file paths and commands
MAIN_TEX: str = os.path.join(os.getcwd(), 'report/main_template.tex')
OUTPUT_TEX: str = os.path.join(os.getcwd(), 'output/main.tex')
TEX_IMAGES_PATH: str = os.path.join(os.getcwd(), 'report/images')
BIRD_IMAGE_PATH: str = os.path.join(TEX_IMAGES_PATH, 'bird_test.jpg')
SAVE_CONF_MATRIX_PATH: str = os.path.join(TEX_IMAGES_PATH, 'output.jpg')
MODEL_PLOT_PATH: str = os.path.join(TEX_IMAGES_PATH, 'model_plot.jpg')
COMMAND: str = 'pdflatex -output-directory ' + os.path.join(os.getcwd(), 'output') + ' ' + OUTPUT_TEX

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
    model_path = os.path.join(TRAINED_MODELS_PATH, CLASSIFIERS[selected_model])

    # Load the selected model
    classifier = load_model(model_path)

    # Process the input image
    processed_img = process(img)

    # Flatten and reshape image input based on the selected model type
    if selected_model in ['Random Forest', 'SVM']:
        classifier_input = flatten_image(processed_img)
        classifier_input = classifier_input.reshape(1, classifier_input.shape[0])
    else:
        # handle image process for cnn model
        pass

    # Load test data
    X_test, y_test = load_flattened_data(DATA_PATH, SPECIES, split='test')

    # Generate respective plots based on the selected model
    if selected_model == 'Random Forest':
        generate_rf_plot(X_test, y_test, MODEL_PLOT_PATH)
    elif selected_model == 'SVM':
        generate_svm_plot(X_test, y_test, MODEL_PLOT_PATH)
    else:
        generate_cnn_plot(X_test, y_test, MODEL_PLOT_PATH)

    # Predict the class of the input image
    predicted_class = classifier.predict(classifier_input)

    # Get probability scores for each class
    prob_scores = classifier.predict_proba(classifier_input)

    # Generate classifier report and confusion matrix
    classifier_report = get_classifier_report(X_test, y_test, SPECIES, model_path, SAVE_CONF_MATRIX_PATH)

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
        'conf_matrix': SAVE_CONF_MATRIX_PATH,
        'accu_plot': MODEL_PLOT_PATH
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

img = Image.open('/home/sarvam/Documents/cs_699/CS_699/data/Blue Jay/47351251.jpg')
generate_report('SVM', img)






    



