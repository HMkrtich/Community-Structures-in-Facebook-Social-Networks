from Unsupervised_models.models import *

def main():
    models = [Models.KMEANS, Models.DBSCAN, Models.MEANSHIFT, Models.GMM, Models.HIERS]

    data = np.array([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0],
                 [1.0, 0.6], [9.0, 11.0], [8.0, 2.0], [10.0, 2.0],
                 [9.0, 3.0]])
    
    UnSup = UnsupservisedModels()

    for model in models:
        print(model)
        model = UnSup.get_model(model, 3)
        model.train(data)
        centroids, labels = model.get_results()
        visualize(data, labels, centroids)

if __name__ == "__main__":
    main()
    
    