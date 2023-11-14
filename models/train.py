from utils import load_flattened_data
from classifier import *

X_train, y_train = load_flattened_data(DATA_PATH, SPECIES, split = 'train')

TRAINED_MODELS_PATH = os.path.join(os.getcwd(), 'models/trained_models')

def main() -> None:
    """Trains Random Forest and SVM classifiers and saves the models.

    Raises:
    - Any exceptions that might occur during training.
    """
    train_RF_classifier(X_train, y_train, TRAINED_MODELS_PATH)
    train_SVM_classifier(X_train, y_train, TRAINED_MODELS_PATH)
    # train_CNN_classifier(X_train, y_train, trained_models_path)



if __name__ == "__main__":
    main()