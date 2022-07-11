from elasticsearch6 import Elasticsearch
from GenericModules.Utilities.JsonFunctions import load_json_file, store_json_file
from datetime import datetime
from dataclasses import dataclass, field, InitVar


@dataclass
class ElasticAPI:
    # INIT ################################################################################################

    host: InitVar[str]
    port: InitVar[int]
    client: Elasticsearch = field(init=False)

    def __post_init__(self, host: str, port: int):
        self.client = Elasticsearch(f'{host}:{port}')

    def __enter__(self):
        return self

    def __exit__(self):
        pass

    def get_info_cluster(self):
        pass

    # RUN QUERY ################################################################################################

    def get_raw_data_from_json_query(self, index, path_json_query, from_=0, size=10000, update_timestamp_query=False):

        str_json_query = load_json_file(path_json_query)
        raw_data = self.client.search(index=index, body=str_json_query, from_=from_, size=size)

        if update_timestamp_query:
            self.update_json_query_from_now(str_json_query, path_json_query)

        return raw_data

    # UPDATE QUERY  ################################################################################################

    # rework this
    @staticmethod
    def update_json_query_from_now(json_query, save_path_query, elastic_utc_format="%Y-%m-%dT%H:%M:%S.%f"):

        if 'must' not in json_query['query']['bool'].keys():
            json_query['query']['bool']['must'] = []

        time_actualised = f'{datetime.now().strftime(elastic_utc_format)}+00:00'
        created = False

        # if (range in utctimestamp) already exist in the query then modify it
        for request in json_query['query']['bool']['must']:
            if type(request) is dict:
                if 'range' in request.keys():
                    if 'utctimestamp' in request['range'].keys():
                        request['range']['utctimestamp'] = {"gt": time_actualised}
                        created = True
                        break

        # else create it
        if not created:
            json_query['query']['bool']['must'].append({'range': {'utctimestamp': {'gt': time_actualised}}})

        # save it
        store_json_file(json_query, save_path_query)
