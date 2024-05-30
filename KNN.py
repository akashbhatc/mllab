import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

class KNN:
    def __init__(self, k, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        if self.distance_metric == 'euclidean':
            distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        elif self.distance_metric == 'manhattan':
            distances = [manhattan_distance(x, x_train) for x_train in self.X_train]

        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]

# Load dataset
data = pd.read_csv(r"C:\Users\Akash\Downloads\glass.csv")
y = data['Type'].values
X = data.drop('Type', axis=1).values

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize KNN with k=3 and Euclidean distance
euclidean = KNN(k=3, distance_metric='euclidean')
euclidean.fit(X_train, y_train)
predictions_euclidean = euclidean.predict(X_test)

# Calculate accuracy for Euclidean distance
accuracy_euclidean = np.sum(predictions_euclidean == y_test) / len(y_test)
print("Euclidean Distance - Predictions:", predictions_euclidean)
print("Euclidean Distance - Accuracy:", accuracy_euclidean)

# Initialize KNN with k=3 and Manhattan distance
manhattan = KNN(k=3, distance_metric='manhattan')
manhattan.fit(X_train, y_train)
predictions_manhattan = manhattan.predict(X_test)

# Calculate accuracy for Manhattan distance
accuracy_manhattan = np.sum(predictions_manhattan == y_test) / len(y_test)
print("Manhattan Distance - Predictions:", predictions_manhattan)
print("Manhattan Distance - Accuracy:", accuracy_manhattan)
