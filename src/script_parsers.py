from scripts.batch_model_train import batch_model_train_parser
from scripts.benchmark_model_speed import benchmark_model_speed_parser
from scripts.dataset_info import dataset_info_parser
from scripts.grid_search_train_and_test import grid_search_parser
from scripts.model_test import model_test_parser
from scripts.model_train import model_train_parser
from scripts.network_info import network_info_parser
from scripts.preprocess_data import preprocess_data_parser

script_parsers = [
    batch_model_train_parser,
    benchmark_model_speed_parser,
    dataset_info_parser,
    grid_search_parser,
    model_test_parser,
    model_train_parser,
    network_info_parser,
    preprocess_data_parser,
]
