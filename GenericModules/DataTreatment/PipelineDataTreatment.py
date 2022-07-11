from DataCleaning import CleanData
from DataPreprocessing import DataPreprocessing


class PipelineDataTreatment:

    def __init__(self):
        pass

    def make_dataset_treatment(self, x_data, y_data=None):
        # ici il faut mettre le y en param pour supprimer paralemement les y
        x_data = CleanData().generic_cleaning_dataframe(x_data, **self.dict_param_clean_data)

        x_data, y_data, numerical_preprocessor, categorical_preprocessor, encoder_y_data = \
            DataPreprocessing().preprocessing_dataset(x_data, y_data, **self.dict_param_preprocess_data)

        return x_data, y_data, numerical_preprocessor, categorical_preprocessor, encoder_y_data
