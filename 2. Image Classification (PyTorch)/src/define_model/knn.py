import numpy as np
from tqdm import tqdm


class KNN:
    """
    A class that imlements the K Nearest Neigbours model.
    
    Attributes:
        k (int): Number of nearest neigbours
        train_x (np.array): Training data
        train_y (np.array): Labels for the training data

    Methods:
        train(X, y):
            Saves the training data.
        predict(X):
            Predicts a class for each element of X.
        evaluate(X, true_y):
            Predicts classes for X and evaluates the accuracy value of the predictions.
    """
    
    def __init__(self, k):
        self.k = k
    
    def train(self, X, y):
        """Saves the training data. Makes predictions based on the data."""
        train_n = X.shape[0]
        self.train_X = X.reshape((train_n, -1))
        self.train_y = y.flatten()
        
    def predict(self, X):
        """Predicts a class for each element of X based on the saved training data."""
        test_n = X.shape[0]
        test_X = X.reshape((test_n, -1))
        pred_y = np.zeros(test_n, dtype=self.train_y.dtype)
        for test_i in tqdm(range(test_n), desc="Prediction: "):
            distances = np.sum(np.abs(self.train_X - test_X[test_i, :]), axis=1)
            k_nearest_idxes = np.argsort(distances)[:self.k]
            k_nearest_preds = self.train_y[k_nearest_idxes]
            unique_preds, count_preds = np.unique(k_nearest_preds, return_counts=True)
            pred_idx = np.argmax(count_preds)
            pred = unique_preds[pred_idx]
            pred_y[test_i] = pred
        return pred_y
    
    def evaluate(self, X, true_y):
        """Predicts classes for X and evaluates the accuracy value of the predictions."""
        pred_y = self.predict(X)
        correct = sum(pred_y == true_y)
        total = X.shape[0]
        return correct / total
