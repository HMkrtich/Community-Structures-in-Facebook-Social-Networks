from enum import Enum
from sklearn.cluster import KMeans, DBSCAN, MeanShift, AgglomerativeClustering, OPTICS
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn import metrics

# Model types
class Models(Enum):
    KMEANS = 1
    DBSCAN = 2
    MEANSHIFT = 3
    GMM = 4
    HIERS = 5

# Unsupervised models
class UnsupservisedModels:
    def __init__(self):
        pass

    def get_model(self, model):
        
        if model == Models.KMEANS:
            return My_KMeans()
        elif model == Models.DBSCAN:
            return MY_DBSCAN()
        # elif model == Models.MEANSHIFT:
        #     return MeanShift()
        # elif model == Models.GMM:
        #     return GaussianMixture()
        # elif model == Models.HIERS:
        #     return AgglomerativeClustering()
        else:
            raise ValueError("Model not found")

# KMeans
class My_KMeans:
    def __init__(self, k):
        self.k = k
        print("KMeans initialized")
        
    def train(self, data):
        print("Training KMeans")
        self.model = KMeans(n_clusters=self.k)

        self.data = data
        self.model.fit(data)

        self.centroids = self.model.cluster_centers_
        self.labels = self.model.labels_

    # Get results
    def get_results(self):
        return self.labels
    
# DBSCAN
class My_DBSCAN:
    def __init__(self, eps, min_samples):
        self.eps = 0.5
        self.min_samples = 3
        # Initialize DBSCAN
        self.model = DBSCAN(eps=eps, min_samples=min_samples)

    def train(self, data):
        self.data = data
        self.model.fit(data)

        self.labels = self.model.labels_

    # Get results
    def get_results(self):
        return self.labels
    
# MeanShift
class My_MeanShift:
    def __init__(self) -> None:
        self.model = MeanShift()

    def train(self, data):
        self.data = data
        self.model.fit(data)

        self.labels = self.model.labels_

    # Get results
    def get_results(self):
        return self.labels
    
# GaussianMixture
class My_GMM:
    def __init__(self, k):
        self.k = k
        self.model = GaussianMixture(self.k)

    def train(self, data):
        self.data = data
        self.model.fit(data)

        self.labels = self.model.predict(data)

    # Get results
    def get_results(self):
        return self.labels
    
# AgglomerativeClustering
class My_Hier:
    def __init__(self, k, affinity, linkage):
        self.k = k
        self.affinity = affinity
        self.linkage = linkage

        self.model = AgglomerativeClustering(n_clusters=self.k, affinity=self.affinity, linkage=self.linkage)

    def train(self, data):
        self.data = data
        self.model.fit(data)

        self.labels = self.model.labels_

    # Get results
    def get_results(self):
        return self.labels
    
# OPTICS
class My_OPTICS:
    def __init__(self, min_samples=5, xi=0.05, min_cluster_size=0.05):
        self.model = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)

    def train(self, data):
        self.data = data
        self.model.fit(data)
        self.labels = self.model.labels_

    def get_results(self):
        return self.labels


# Helpers
    
# Visualize the results
def visualize(data, labels):
    # Create a colormap
    num_labels = len(set(labels))
    colormap = cm.get_cmap('rainbow', num_labels)  # Use 'rainbow' or any other colormap

    # Draw the nodes - color by label
    color_map = [colormap(label / num_labels) for label in labels]

    plt.scatter(data[:, 0], data[:, 1], c=color_map)

    # Draw a circle for each cluster
    for label in set(labels):
        cluster_points = [data[i] for i in range(len(data)) if labels[i] == label]
        centroid = np.mean(cluster_points, axis=0)
        furthest_distance = max(np.linalg.norm(point - centroid) for point in cluster_points)
        circle = plt.Circle(centroid, furthest_distance, fill=False)
        plt.gca().add_patch(circle)

    plt.show()

# Extract the data from the text
def extract_data(text):
    data = []
    for line in text:
        line = line.split()
        data.append([float(line[0]), float(line[1])])
    return np.array(data)
    
# Test labels
def test_labels(labels, test_labels):
    print("Adjusted Rand index:", metrics.adjusted_rand_score(test_labels, labels))
    print("Normalized Mutual Information:", metrics.normalized_mutual_info_score(test_labels, labels))
    print("Homogeneity:", metrics.homogeneity_score(test_labels, labels))
    print("Completeness:", metrics.completeness_score(test_labels, labels))
    print("V-measure:", metrics.v_measure_score(test_labels, labels))
    print("Silhouette Coefficient:", metrics.silhouette_score(data, labels))

def read_labels(filename):
    labels = {}
    with open(filename, 'r') as file:
        for i, line in enumerate(file):
            words = line.split()
            for word in words[1:]:  # Skip the circle name
                labels[int(word)] = i  # Assign the label
    
    final_labels = []
    for i in range(len(labels)):
        final_labels.append(labels[i + 1])
    return final_labels

# # Usage:
# labels = read_labels("../facebook/0.circles")
# print(labels)