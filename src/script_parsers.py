from scripts.dataset_info import dataset_info_parser
from scripts.model_train import model_train_parser
from scripts.network_info import network_info_parser
from scripts.preprocess_data import preprocess_data_parser

script_parsers = [
    dataset_info_parser,
    model_train_parser,
    network_info_parser,
    preprocess_data_parser,
]
