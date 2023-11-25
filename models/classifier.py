import pickle
from abc import abstractmethod
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from tensorflow import keras

from mlxtend.plotting import heatmap
from models.utils import save_model
import models.utils as utils

# List of species names used in classification
SPECIES = ['Blue Jay', 'Grey Heron', 'Indian Peafowl', 'Little Egret', 'Red-Vented Bulbul']

# Mapping of classifier names to their respective types
CLASSIFIERS = {
    'Random Forest': 'Random Forest',  # Classifier type: Random Forest
    'SVM': 'SVM',  # Classifier type: Support Vector Machine
    'CNN': 'CNN'  # Classifier type: Convolutional Neural Network
}


class MyClassifier:
    # Path to the directory where trained models will be stored
    TRAINED_MODELS_PATH = Path('.') / 'models' / 'trained_models'

    def __init__(self, dir_data, model_type):
        """
        Initialize the classifier.

        Args:
        - dir_data (str): Directory path to the dataset.
        - model_type (str): Type of the model ('Random Forest', 'SVM', 'CNN').
        """
        self.DATA_PATH = Path(dir_data)
        self.BATCH_SIZE = 32
        self.IMAGE_SIZE = (512, 512)
        self.name = model_type
        self.model = None
        self.model_type = model_type
        if model_type == 'Random Forest' or model_type == 'SVM':
            # Load flattened data for certain model types
            self.X_train, self.y_train = utils.load_flattened_data(dir_data, SPECIES, split='train')
            self.X_test, self.y_test = utils.load_flattened_data(dir_data, SPECIES, split='test')
            self.CLASSES = SPECIES
        else:
            # Load image datasets for other model types
            self.ds_train, self.ds_val = keras.utils.image_dataset_from_directory(
                directory=self.DATA_PATH / 'train',
                label_mode='categorical',
                batch_size=self.BATCH_SIZE,
                image_size=self.IMAGE_SIZE,
                validation_split=0.2,
                subset='both',
                seed=4,
            )
            self.ds_test = keras.utils.image_dataset_from_directory(
                directory=self.DATA_PATH / 'test',
                label_mode='categorical',
                batch_size=self.BATCH_SIZE,
                image_size=self.IMAGE_SIZE,
            )
            self.CLASSES = self.ds_train.class_names
        self.model_path = MyClassifier.TRAINED_MODELS_PATH / CLASSIFIERS[model_type]

    @abstractmethod
    def train(self, save_model_path) -> None:
        """
        Train the classifier and save the model.

        Args:
        - save_model_path (str): Path to save the trained model.
        """
        pass

    @abstractmethod
    def test(self) -> tuple:
        """
        Generate predictions on the test data.

        Returns:
        - tuple: Tuple containing true and predicted values.
        """
        pass

    @staticmethod
    @abstractmethod
    def load(model_path):
        """
        Load a pre-trained model.

        Args:
        - model_path (str): Path to the pre-trained model.

        Returns:
        - object: The loaded model object.
        """
        pass

    @staticmethod
    @abstractmethod
    def predict(model_path, X):
        """
        Generate predictions using a pre-trained model.

        Args:
        - model_path (str): Path to the pre-trained model.
        - X : Input data for prediction.

        Returns:
        - prediction: The prediction made by the model.
        """
        pass

    @staticmethod
    def _get_classifier_report(y_true, y_pred, target_labels, save_conf_matrix_path):
        """
        Generate a classification report and save the confusion matrix plot.

        Args:
        - y_true: True labels.
        - y_pred: Predicted labels.
        - target_labels: Labels for the classification.
        - save_conf_matrix_path (str): Path to save the confusion matrix plot.

        Returns:
        - report: The classification report.
        """
        report = classification_report(y_true, y_pred)  # Generate the classification report

        # Generate and save the confusion matrix plot
        conf_mat = confusion_matrix(y_true, y_pred)
        heatmap(conf_mat, column_names=target_labels, row_names=target_labels, figsize=(20, 20), cmap="BuPu")

        fig = plt.gcf()  # Get the current figure
        plt.figure()
        fig.savefig(save_conf_matrix_path)  # Save the confusion matrix plot

        return report  # Return the classification report


class ClassifierSVM(MyClassifier):
    def __init__(self, dir_data):
        """
        Initialize the SVM classifier.

        Args:
        - dir_data (str): Directory path to the dataset.
        """
        super().__init__(dir_data, 'SVM')

    def train(self, save_model_path=None) -> None:
        """
        Train the SVM classifier and save the model.

        Args:
        - save_model_path (str): Path to save the trained model.
        """
        classifier = SVC(kernel='linear', probability=True)  # Initialize SVM classifier
        classifier.fit(self.X_train, self.y_train)  # Train the classifier
        save_model(save_model_path if save_model_path else self.model_path, classifier)  # Save the trained model
        self.model = classifier
        self.test()

    def test(self) -> tuple:
        """
        Generate predictions on the test data.

        Returns:
        - tuple: Tuple containing true and predicted values.
        """
        y_bar = self.model.predict(self.X_test)

        C_values = [0.001, 0.01, 0.1, 1, 10, 100]
        accuracy_values = []

        # Iterating through different C values
        for C in C_values:
            svm_model = SVC(kernel='linear', C=C, random_state=42)
            svm_model.fit(self.X_test, self.y_test)
            accuracy = svm_model.score(self.X_test, self.y_test)
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
        fig.savefig(self.model_path / 'accuracy.png', bbox_inches='tight')

        with open(self.model_path / 'test_run.bin', 'wb') as f:
            pickle.dump((self.y_test, y_bar), f)
        return self.y_test, y_bar

    @staticmethod
    def load(model_path):
        """
        Load a pre-trained SVM model.

        Args:
        - model_path (str): Path to the pre-trained model.

        Returns:
        - object: The loaded model object.
        """
        return utils.load_model(model_path)  # Load the trained model

    @staticmethod
    def predict(model_path, img):
        """
        Generate predictions using a pre-trained SVM model.

        Args:
        - model_path (str): Path to the pre-trained model.
        - img: Input data for prediction.

        Returns:
        - prediction: The prediction made by the model.
        """
        classifier = ClassifierSVM.load(model_path)
        classifier_input = utils.flatten_image(img)
        classifier_input = classifier_input.reshape(1, classifier_input.shape[0])
        return classifier.predict(classifier_input), classifier.predict_proba(classifier_input)

    @staticmethod
    def get_classifier_report(model_path):
        """
        Generate a classification report using the SVM classifier.

        Args:
        - model_path (str): Path to the model.

        Returns:
        - report: The classification report.
        """
        with open(model_path / 'test_run.bin', 'rb') as f:
            (y_true, y_pred) = pickle.load(f)
        if (model_path / 'cls_report.txt').exists() and (model_path / 'conf.jpg').exists():
            with open(model_path / 'cls_report.txt', 'r') as f:
                return f.read()
        else:
            report = MyClassifier._get_classifier_report(y_true, y_pred, SPECIES, model_path / 'conf.jpg')
            with open(model_path / 'cls_report.txt', 'w') as f:
                f.write(report)
            return report


class ClassifierRF(MyClassifier):
    def __init__(self, dir_data):
        """
        Initialize the Random Forest classifier.

        Args:
        - dir_data (str): Directory path to the dataset.
        """
        super().__init__(dir_data, 'Random Forest')

    def train(self, save_model_path=None) -> None:
        """
        Train the Random Forest classifier and save the model.

        Args:
        - save_model_path (str): Path to save the trained model.
        """
        classifier = RandomForestClassifier()  # Initialize Random Forest classifier
        classifier.fit(self.X_train, self.y_train)  # Train the classifier
        save_model(save_model_path if save_model_path else self.model_path, classifier)  # Save the trained model
        self.model = classifier
        self.test()

    def test(self) -> tuple:
        """
        Generate predictions on the test data.

        Returns:
        - tuple: Tuple containing true and predicted values.
        """
        y_bar = self.model.predict(self.X_test)

        accuracy_values = []

        # Training Random Forest with different numbers of trees
        for n_trees in range(1, 101):
            rf_model = RandomForestClassifier(n_estimators=n_trees, random_state=42)
            rf_model.fit(self.X_test, self.y_test)
            accuracy = rf_model.score(self.X_test, self.y_test)
            accuracy_values.append(accuracy)

        # Plotting accuracy over the number of trees
        plt.plot(range(1, 101), accuracy_values, marker='o')
        plt.title('Random Forest Accuracy over Number of Trees')
        plt.xlabel('Number of Trees')
        plt.ylabel('Accuracy')

        # Save the generated plot
        fig = plt.gcf()
        plt.figure()
        fig.savefig(self.model_path / 'accuracy.png', bbox_inches='tight')

        with open(self.model_path / 'test_run.bin', 'wb') as f:
            pickle.dump((self.y_test, y_bar), f)
        return self.y_test, y_bar

    @staticmethod
    def load(model_path):
        """
        Load a pre-trained Random Forest model.

        Args:
        - model_path (str): Path to the pre-trained model.

        Returns:
        - object: The loaded model object.
        """
        return utils.load_model(model_path)  # Load the trained model

    @staticmethod
    def predict(model_path, img):
        """
        Generate predictions using a pre-trained Random Forest model.

        Args:
        - model_path (str): Path to the pre-trained model.
        - img: Input data for prediction.

        Returns:
        - prediction: The prediction made by the model.
        """
        classifier = ClassifierRF.load(model_path)
        classifier_input = utils.flatten_image(img)
        classifier_input = classifier_input.reshape(1, classifier_input.shape[0])
        return classifier.predict(classifier_input), classifier.predict_proba(classifier_input)

    @staticmethod
    def get_classifier_report(model_path):
        """
        Generate a classification report using the Random Forest classifier.

        Args:
        - model_path (str): Path to the model.

        Returns:
        - report: The classification report.
        """
        with open(model_path / 'test_run.bin', 'rb') as f:
            (y_true, y_pred) = pickle.load(f)
        if (model_path / 'cls_report.txt').exists() and (model_path / 'conf.jpg').exists():
            with open(model_path / 'cls_report.txt', 'r') as f:
                return f.read()
        else:
            report = MyClassifier._get_classifier_report(y_true, y_pred, SPECIES, model_path / 'conf.jpg')
            with open(model_path / 'cls_report.txt', 'w') as f:
                f.write(report)
            return report


class ClassifierCNN(MyClassifier):
    def __init__(self, dir_data):
        """
        Initialize the CNN classifier.

        Args:
        - dir_data (str): Directory path to the dataset.
        """
        super().__init__(dir_data, 'CNN')

    def train(self, epochs=5, save_model_path=None) -> None:
        """
        Train the CNN classifier and save the model.

        Args:
        - epochs (int): Number of epochs for training.
        - save_model_path (str): Path to save the trained model.
        """
        input_shape = self.IMAGE_SIZE + (3,)
        base_model = keras.applications.EfficientNetV2M(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape,
        )
        base_model.trainable = False
        inputs = keras.Input(shape=input_shape)
        x = base_model(inputs, training=False)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dense(len(self.CLASSES))(x)
        outputs = keras.layers.Softmax()(x)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                      loss=keras.losses.CategoricalCrossentropy(),
                      metrics=[keras.metrics.CategoricalAccuracy()])
        history = model.fit(self.ds_train, epochs=epochs, validation_data=self.ds_val)

        self.model = model
        self.model.save(self.model_path)

        self.plot(history)
        self.test()

        with open(self.model_path / 'labels.bin', 'wb') as f:
            pickle.dump(self.CLASSES, f)
        with open(self.model_path / 'history.bin', 'wb') as f:
            pickle.dump(history, f)

        return history

    def plot(self, history):
        """
        Generate plots for accuracy and loss.

        Args:
        - history: The training history.
        """
        # accuracy plot
        plt.plot(history.history['categorical_accuracy'])
        plt.plot(history.history['val_categorical_accuracy'])
        plt.title('accuracy plot')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(self.model_path / 'accuracy.png', bbox_inches='tight')
        plt.close()

        # loss plot
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('loss plot')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(self.model_path / 'loss.png', bbox_inches='tight')
        plt.close()

    def test(self):
        """
        Generate predictions on the test data.

        Returns:
        - tuple: Tuple containing true and predicted values.
        """
        y = np.concatenate([y for x, y in self.ds_test], axis=0)
        y = np.argmax(y, axis=1)

        pred = self.model.predict(self.ds_test)
        y_bar = np.argmax(pred, axis=1)

        with open(self.model_path / 'test_run.bin', 'wb') as f:
            pickle.dump((y, y_bar), f)
        return y, y_bar

    @staticmethod
    def load(model_path) -> keras.Model:
        """
        Load a pre-trained CNN model.

        Args:
        - model_path (str): Path to the pre-trained model.

        Returns:
        - object: The loaded model object.
        """
        return keras.models.load_model(model_path)

    @staticmethod
    def predict(model_path, img):
        """
        Generate predictions using a pre-trained CNN model.

        Args:
        - model_path (str): Path to the pre-trained model.
        - img: Input data for prediction.

        Returns:
        - prediction: The prediction made by the model.
        """
        model = ClassifierCNN.load(model_path)
        with open(Path(model_path) / 'labels.bin', 'rb') as f:
            classes = pickle.load(f)
        np_image = np.array(img).astype('float32') / 255
        np_image = np.expand_dims(np_image, axis=0)
        preds = model.predict(np_image)
        preds_label = np.argmax(preds, axis=1)
        return [classes[pred_label] for pred_label in preds_label], preds

    @staticmethod
    def get_classifier_report(model_path):
        """
        Generate a classification report using the CNN classifier.

        Args:
        - model_path (str): Path to the model.

        Returns:
        - report: The classification report.
        """
        model_path = Path(model_path)
        with open(model_path / 'test_run.bin', 'rb') as f:
            (y_true, y_pred) = pickle.load(f)
        with open(model_path / 'labels.bin', 'rb') as f:
            classes = pickle.load(f)

        if (model_path / 'cls_report.txt').exists() and (model_path / 'conf.jpg').exists():
            with open(model_path / 'cls_report.txt', 'r') as f:
                return f.read()
        else:
            report = MyClassifier._get_classifier_report(y_true, y_pred, classes, model_path / 'conf.jpg')
            with open(model_path / 'cls_report.txt', 'w') as f:
                f.write(report)
            return report
