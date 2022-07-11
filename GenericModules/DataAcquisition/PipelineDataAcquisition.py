from ElasticAPI import ElasticAPI
from PostTreatmentRequest import PostTreatmentQuery


class PipelineDataAcquisition:

    def __init__(self, host, port, list_additional_features_to_extract):
        self.client = ElasticAPI(host, port)
        self.pt_query = PostTreatmentQuery(list_additional_features_to_extract, output_type_in_dict_df=True)

    # EVENTS ################################################################################################

    def get_events_by_observations(self, index_cluster, path_json_query, update_timestamp_query):
        # get data and update query
        raw_data = self.client.get_raw_data_from_json_query(index_cluster, path_json_query,
                                                            update_timestamp_query=update_timestamp_query)

        # sort and extract observation data
        dict_df = self.pt_query.extract_events__sort_it_by_observation(raw_data)

        return dict_df
