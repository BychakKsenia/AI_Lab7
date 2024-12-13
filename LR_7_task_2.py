import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# Завантаження набору даних Iris
iris = load_iris()
X = iris.data
y = iris.target

# Використовуємо PCA для зменшення кількості вимірів до 2 для візуалізації
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Кількість кластерів (3 класи квітів: Setosa, Versicolour і Virginica)
num_clusters = 3

# Виконуємо кластеризацію KMeans
kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10, random_state=42)
kmeans.fit(X)

# Передбачення для всього набору даних
y_kmeans = kmeans.predict(X)

# Візуалізація результатів кластеризації
plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap=plt.cm.Paired, s=80, edgecolor='black')
plt.title('Кластери набору даних Iris (KMeans)')

# Додаємо центри кластерів на графік
cluster_centers = kmeans.cluster_centers_
cluster_centers_pca = pca.transform(cluster_centers)
plt.scatter(cluster_centers_pca[:, 0], cluster_centers_pca[:, 1],
            marker='x', s=210, linewidths=4, zorder=12, facecolors='red')

plt.xlabel('Головна компонента 1')
plt.ylabel('Головна компонента 2')
plt.show()
