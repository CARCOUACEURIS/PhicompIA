
def load_unsupervised_model(name_unsupervised_model):
    if name_unsupervised_model is not None:
        name_unsupervised_model = name_unsupervised_model.replace('_', '').replace(' ', '').lower()

    if name_unsupervised_model in ['dbscan', None]:
        from sklearn.cluster import DBSCAN
        return DBSCAN

    if name_unsupervised_model in ['hdbscan']:
        from hdbscan.hdbscan_ import HDBSCAN
        return HDBSCAN

    if name_unsupervised_model in ['kmeans', 'kmean']:
        from sklearn.cluster import KMeans
        return KMeans