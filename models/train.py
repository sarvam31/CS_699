from utils import *
from classifier import *

X_train, y_train = utils.load_flattened_data(DATA_PATH, SPECIES, split = 'train')

TRAINED_MODELS_PATH = os.path.join(os.getcwd(), 'models/trained_models')

def main():
    train_RF_classifier(X_train, y_train, TRAINED_MODELS_PATH)
    train_SVM_classifier(X_train, y_train, TRAINED_MODELS_PATH)
    # train_CNN_classifier(X_train, y_train, trained_models_path)


if __name__ == "__main__":
    main()