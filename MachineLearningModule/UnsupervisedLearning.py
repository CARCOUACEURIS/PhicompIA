from sklearn.metrics import silhouette_score,silhouette_samples
from LibraryModelsAndObjects.UnsupervisedModels import load_unsupervised_model


class UnsupervisedLearning:

    def __init__(self, cluster_program_name, dict_param=None):

        self.cluster_program = load_unsupervised_model(cluster_program_name)

        self.dict_param = dict_param
        if self.dict_param is None:
            self.dict_param = {}

    def fit(self, data):
        self.cluster_program(**self.dict_param).fit(data)

    def fit_predict(self, data):
        labels = self.cluster_program(**self.dict_param).fit_predict(data)
        return labels

    @staticmethod
    def silhouette_score(data, labels):
        """
            Compute the mean Silhouette Coefficient of all samples.

            This measure has a range of [-1, 1].
            Silhouette coefficients (as these values are referred to as) near +1 indicate that the sample is far away
            from the neighboring clusters. A value of 0 indicates that the sample is on or very close to the decision
            boundary between two neighboring clusters and negative values indicate that those samples might have been
            assigned to the wrong cluster.
        """
        return silhouette_score(data, labels)

    @staticmethod
    def silhouette_samples(self, data, labels):
        """
            Compute the silhouette scores for each sample.
        """
        return silhouette_samples(data, labels)