from dataclasses import dataclass


@dataclass
class LoadPreprocessorObjects:

    @staticmethod
    def load_scaler(scaler):
        if scaler is not None:
            scaler = scaler.replace('_', '').replace(' ', '').lower()

        if scaler in ['robust', 'robustscaler', None]:
            from sklearn.preprocessing import RobustScaler
            return RobustScaler(quantile_range=(25.0, 75.0))

        if scaler in ['standard', 'standardscaler']:
            from sklearn.preprocessing import StandardScaler
            return StandardScaler()

        if scaler in ['minmax', 'minmaxscaler']:
            from sklearn.preprocessing import MinMaxScaler
            return MinMaxScaler()

    @staticmethod
    def load_encoder(encoder):
        if encoder is not None:
            encoder = encoder.replace('_', '').replace(' ', '').lower()

        if encoder in ['onehot', 'onehotencoder']:
            from sklearn.preprocessing import OneHotEncoder
            return OneHotEncoder()
        if encoder in ['label', 'labelencoder']:
            from sklearn.preprocessing import LabelEncoder
            return LabelEncoder()

    @staticmethod
    def load_imputer(imputer):
        if imputer is not None:
            imputer = imputer.replace('_', '').replace(' ', '').lower()

        if imputer in ['knn', 'knnimputer']:
            from sklearn.impute import KNNImputer
            return KNNImputer()

        if imputer in ['simple', 'simpleimputer', None]:  # passer None si possible
            from sklearn.impute import SimpleImputer
            return SimpleImputer(strategy='constant', fill_value='null')
