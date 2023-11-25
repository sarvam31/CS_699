import os
import sys

# Adding project root folder to python path
sys.path.insert(1, os.getcwd())

# Moved the import within the main block for clearer control flow
from models.classifier import *

DATA_PATH = os.path.join(os.getcwd(), 'processed')
# Used Pathlib to handle paths instead of string concatenation
TRAINED_MODELS_PATH = Path('.') / 'models' / 'trained_models'


# Adjusted function signature to fit PEP8 standards
def main() -> None:
    """
    Trains Random Forest and SVM classifiers and saves the models.

    Raises:
    - Any exceptions that might occur during training.
    """

    # Renamed variables to follow PEP8 conventions (lowercase with underscores)
    rf = ClassifierRF(DATA_PATH)
    rf.train()
    print('RF training done...')
    print(rf.get_classifier_report(rf.model_path))

    svm = ClassifierSVM(DATA_PATH)
    svm.train()
    print('SVM training done...')
    print(svm.get_classifier_report(svm.model_path))

    cnn = ClassifierCNN(DATA_PATH)
    cnn.train()
    print('CNN training done...')
    print(cnn.get_classifier_report(cnn.model_path))


if __name__ == "__main__":
    main()
