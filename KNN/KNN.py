import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def euclidean_distance(p,q):
    return np.sqrt(np.sum((p-q)**2))

class KNearestClassifier:
    def __init__(self,k=5):
        self.k=k

    #Fit method 
    def fit(self,X,y):
        self.X_train=X
        self.y_train=y
    # predict methpd
    def predict(self,X):
        predictions=[]
        for x in X:
            distances=[]
            for i in self.X_train:
                distance=euclidean_distance(x,i)
                distances.append(distance)
            k_indices=np.argsort(distances)[:self.k]
            k_labels=[]
            for i in k_indices:
                k_labels.append(self.y_train[i])
            most=Counter(k_labels).most_common()
            predictions.append(most[0][0])
        return predictions