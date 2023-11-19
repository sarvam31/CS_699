from .utils import load_flattened_data
from .classifier import *
import os

DATA_PATH = os.path.join(os.getcwd(), 'processed')
TRAINED_MODELS_PATH = Path('.') / 'models' / 'trained_models'


def main() -> None:
    """Trains Random Forest and SVM classifiers and saves the models.

    Raises:
    - Any exceptions that might occur during training.
    """
    rf = ClassifierRF(DATA_PATH)
    rf.train()
    print(rf.get_classifier_report(rf.model_path))

    svm = ClassifierSVM(DATA_PATH)
    svm.train()
    print(svm.get_classifier_report(svm.model_path))

    cnn = ClassifierCNN(DATA_PATH)
    cnn.train()
    print(cnn.get_classifier_report(cnn.model_path))


if __name__ == "__main__":
    main()
