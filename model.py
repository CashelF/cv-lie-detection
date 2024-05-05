import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.models import Sequential, load_model


class NeuralNetworkModel:
    def __init__(self, csv_file, model_file=None):
        self.dataset = pd.read_csv(csv_file)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.model_file = model_file

    def preprocess_data(self):
        X = self.dataset.drop(columns=['Deception', 'Video'])  # Features (input variables)
        y = self.dataset['Deception']  # Target variable (labels)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def build_model(self):
        self.model = Sequential([
            Dense(64, activation='relu', input_shape=(self.X_train.shape[1],)),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train_model(self, epochs=10, batch_size=32, validation_split=0.1, save_model_file=None):
        history = self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
        if save_model_file:
            self.model.save(save_model_file)

    def evaluate_model(self):
        test_loss, test_accuracy = self.model.evaluate(self.X_test, self.y_test)
        print("Test Loss:", test_loss)
        print("Test Accuracy:", test_accuracy)

    def load_model(self):
        if self.model_file:
            self.model = load_model(self.model_file)

    def predict(self, X):
        self.predictions = self.model.predict(X)
        return self.predictions

    def main(self, load_saved_model=False):
        if load_saved_model:
            self.load_model()
        else:
            self.preprocess_data()
            self.build_model()
            self.train_model()
            self.evaluate_model()


# Example usage:
if __name__ == "__main__":
    model_file = 'trained_model.h5'
    model = NeuralNetworkModel('all_videos_metrics.csv', model_file)
    model.main(load_saved_model=True)  # Load the pre-trained model
    input_data = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
    input_data = np.array(input_data).reshape(1, -1)
    predictions = model.predict(input_data)  
    print("Predictions:", predictions)