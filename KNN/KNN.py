import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def euclidean_distance(p,q):
    distance=np.sqrt(np.sum((p-q)**2))
    return distance

class KNearestClassifier:
    def __init__(self,k=5):
        self.k=k
    def fit(self,X,y):
        self.X_train=X
        self.y_train=y

    def predict(self,X):
        predictions=[self._predict(x) for x in X]
        return predictions
    
    def _predict(self,x):
        distances=[euclidean_distance(x,x_train) for x_train in self.X_train]

        k_indices=np.argsort(distances)[:self.k]
        k_labels=[self.y_train[i] for i in k_indices]

        most=Counter(k_labels).most_common()
        return most[0][0]
