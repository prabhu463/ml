# 9 Naive Bayes Classifier

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

import matplotlib.pyplot as plt

# Load data
X, y = datasets.fetch_olivetti_faces(return_X_y=True)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Train
model = GaussianNB()
model.fit(X_train, y_train)

# Predict & evaluate
y_pred = model.predict(X_test)


# Visualize
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(X_test[i].reshape(64, 64), cmap='gray')
    plt.title(y_pred[i])
    plt.axis('off')
plt.show()