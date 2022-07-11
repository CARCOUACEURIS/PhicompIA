from dataclasses import dataclass


class Production:

    # SETUP - LOAD OBJECTS & MODELS ##################################################################################

    def load_fitted_preprocessor_objects(self):
        pass

    def load_fitted_supervised_model(self):
        pass

    def create_reinforcement_learning_model(self):
        pass

    def load_reinforcement_learning_model(self):
        pass

    # RUN PROGRAM ###################################################################################################

    def run_one_loop(self):

        # API - get
        event = self.get_event_from_elastic_cluster() # 1

        # data cleaning / data preprocessing
        event_preprocessed = self.clean_and_preprocess_event(event) # 2

        # CLF
        alert = self.classify_event(event_preprocessed) # 3

        # API -post
        self.store_alert(alert) # 4

        # RL - reward
        reward = self.calculate_reward_of_last_action() # 5

        # RL - calibrate
        self.calibrate_model(reward) # 6

        # RL - determine
        action = self.determine_action_to_execute(alert) #7

        # API - post
        self.execute_action(action) # 8
        self.store_action(action) # 8 bis

    # API ############################################################################################################

    def get_event_from_elastic_cluster(self):
        pass

    def store_alert(self):
        pass

    def store_action(self):
        pass

    def execute_action(self):
        pass

    # CLEANING PREPROCESSING #########################################################################################

    def clean_and_preprocess_event(self):
        pass

    # ALERTS CLASSIFIER  #############################################################################################

    def classify_event(self):
        pass

    # REINFORCEMENT LEARNING #########################################################################################

    def calculate_reward_of_last_action(self):
        pass

    def calibrate_model(self):
        pass

    def determine_action_to_execute(self):
        pass







