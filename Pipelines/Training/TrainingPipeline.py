from dataclasses import dataclass

from GenericModules.Utilities.FileManipulation import open_pickle_object
from GenericModules.DataAcquisition.PipelineDataAcquisition import  PipelineDataAcquisition
from GenericModules.DataTreatment.PipelineDataTreatment import PipelineDataTreatment
from MachineLearningModule.UnsupervisedLearning import UnsupervisedLearning

@dataclass
class TrainingClassifier:

    name_or_path_classifier:str
    credentials_cluster: dict

    def __post_init__(self):
        self.credentials_cluster = {'host': '', 'port': ''}
        self.data_acquisition = PipelineDataAcquisition(self.host, self.port, self.list_additional_features_to_extract)
        self.data_treatment  =PipelineDataTreatment(dict_param_clean_data, dict_param_preprocess_data)



    def main_pipeline(self):

        # ok
        x_data = self.data_acquisition.get_events_by_observations(self.index_cluster, self.path_json_query,
                                                                self.update_timestamp_query)

        # ok
        x_data_preprocessed,__, encoder, scaler, imputer = self.data_treatment.make_dataset_treatment(x_data, None)


        # optimisation with A.G or baesian naives
        y_data = UnsupervisedLearning('',{}).fit_predict(x_data_preprocessed)

        ###############################################

        # optimisation with A.G or baesian naives
        # load model
        fitted_classifier = self.fit_classifier(classifier, data_preprocessed, labels)

        self.store_models_and_objects(fitted_classifier, encoder, scaler, imputer)

    # LOAD OBJECTS ###################################################################################################

    # need to move in load_model_sup_learning : load from a path
    def load_classifier(self):

        #  from a path : pre-fitted classifier
        if self.path_classifier is not None:
            try:
                return open_pickle_object(self.path_classifier)
            except FileNotFoundError as err:
                self.report['load_classifier'] = f'{FileNotFoundError} : {err}'

        # from a library :  non-fitted classifier
        return LoadObjectsForSupervisedLearning().load_supervised_model(self.name_classifier)


    # API ############################################################################################################

    # data acquisition
    def get_dataset_of_events_from_elastic_cluster(self):


    def store_models_and_objects(self):
        # - store_classifier
        # - store encoder
        # - store scaler
        pass

    # DATA CLEANING / PREPROCESSING ##################################################################################

    def clean_and_preprocess_dataset_of_events(self):
        # return data, encoder scaler, imputer if necessary
        pass

    # UNSUPERVISED LEARNING ##################################################################################

    def create_labels_from_clustering(self):
        pass

    # SUPERVISED LEARNING ##################################################################################

    def fit_classifier(self):
        pass
