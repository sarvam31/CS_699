import os
import sys
import subprocess

sys.path.insert(1, os.getcwd())

from models.train import TRAINED_MODELS_PATH
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


def generate_report(selected_model, img_file):
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
    img = Image.open(img_file)
    processed_img = process(img)

    model = models[selected_model]
    predicted_class, prob_scores = model.predict(model_path, processed_img)
    print(predicted_class, prob_scores, sep='\n')

    # Generate classifier report and confusion matrix
    classifier_report = model.get_classifier_report(model_path)
    print(classifier_report)

    # Load the LaTeX main template file
    tex_code = load_file(MAIN_TEX)

    # Data dictionary to replace placeholders in the LaTeX template
    data = {
        'modelpredictedspecies': predicted_class[0],
        'classifiername': selected_model,
        'probscore1': "{:.2f}".format(prob_scores[0][0]),
        'probscore2': "{:.2f}".format(prob_scores[0][1]),
        'probscore3': "{:.2f}".format(prob_scores[0][2]),
        'probscore4': "{:.2f}".format(prob_scores[0][3]),
        'probscore5': "{:.2f}".format(prob_scores[0][4]),
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
    else:
        return predicted_class



if __name__ == '__main__':
    img_path = Path('data/Little Egret/61001311.jpg')
    with open(img_path, 'rb') as f:
        generate_report('Random Forest', f)
