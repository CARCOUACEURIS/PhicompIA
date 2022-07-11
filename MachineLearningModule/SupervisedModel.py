from sklearn.model_selection import cross_validate, StratifiedKFold
from statistics import stdev

from GenericModules.Utilities.FileManipulation import store_object_as_pickle, create_directory
from LibraryModelsAndObjects.SupervisedModelsObjects import load_supervised_model, load_cross_validation_object


class SupervisedLearning:

    def __init__(self, name_model, classification=True, dict_param_model=None):

        if dict_param_model is None:
            dict_param_model = {}

        self.name_model = name_model
        self.model = self.load_supervised_model(path_model, name_model, classification, dict_param_model)

    @staticmethod
    def load_supervised_model(name_model, classification, dict_param_model):

        if path_model is None:
            return load_supervised_model(name_model, classification)(**dict_param_model)
        else:
            return load_pickle_object()

    # BASC FITTING  #################################################################################################

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    # CROSS VALIDATE FITTING  #####################################################################################

    def repeated_cross_validate_fitting(self, X_train, y_train, repeat_experience=10, nb_cv=5,
                                        type_cv_or_name_type_cv='StratifiedKFold', metric_scoring='accuracy',
                                        save_path_estimator=None):

        type_cv_or_name_type_cv = load_cross_validation_object(type_cv_or_name_type_cv)

        if repeat_experience is None:
            repeat_experience = 1

        if save_path_estimator is not None:
            create_directory(save_path_estimator)

        list_dict_results = []
        for counter in range(repeat_experience):
            # random_state = random.randint(1, 5 * repeat_experience),
            list_dict_results.append(self.cross_validate_fitting(X_train, y_train, nb_cv, type_cv_or_name_type_cv,
                                                                 metric_scoring, save_path_estimator, counter))
        return list_dict_results

    def cross_validate_fitting(self, X_train, y_train, nb_cv=5, name_type_cv_or_type_cv=StratifiedKFold,
                               scoring='accuracy', path_store_estimator=None, counter_repeat=None):

        return_estimator = False if path_store_estimator is None else True

        type_cv = self.objects_loader.load_cross_validation_object(name_type_cv_or_type_cv)
        cross_val = nb_cv
        if type_cv is not None:
            cross_val = type_cv(nb_cv, shuffle=True)

        dict_results = cross_validate(self.model, X=X_train, y=y_train, cv=cross_val, scoring=(scoring, 'r2'),
                                      return_train_score=True, return_estimator=return_estimator)
        print(scoring)
        for i in dict_results:
            print(f'{i} : {dict_results[i]}')
        dict_results = self.treatment_results_cross_validate(dict_results)

        if path_store_estimator is not None:
            self.store_list_estimators_object(dict_results.pop('list_estimator'),
                                              f'{path_store_estimator}_{counter_repeat}')

        dict_results['name_model'] = f'{self.name_model}_{counter_repeat}'
        return dict_results

    @staticmethod
    def treatment_results_cross_validate(dict_results):

        # train score
        dict_results['train_score_mean'] = dict_results['train_score'].mean()
        dict_results['st_dev_train_score'] = stdev(dict_results['train_score'])

        # validate score
        dict_results['validate_score'] = dict_results.pop('test_score')
        dict_results['validate_score_mean'] = dict_results['validate_score'].mean()
        dict_results['st_dev_validate_score'] = stdev(dict_results['validate_score'])

        return dict_results
