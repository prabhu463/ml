# Program 3 (PCA)

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Load dataset
iris = load_iris()

# Standardize data
X_scaled = StandardScaler().fit_transform(iris.data)

# Apply PCA
X_pca = PCA(n_components=2).fit_transform(X_scaled)


# Plot
plt.figure(figsize=(8, 6))

for i, color, label in zip(range(3), ['r', 'g', 'b'], iris.target_names):
    plt.scatter(
        X_pca[iris.target == i, 0],
        X_pca[iris.target == i, 1],
        c=color,
        alpha=0.6,
        label=label
    )

plt.title("PCA of Iris Dataset")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend(title="Classes")
plt.grid()

plt.show()