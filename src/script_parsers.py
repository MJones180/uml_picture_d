from scripts.batch_model_test import batch_model_test_parser
from scripts.batch_model_train import batch_model_train_parser
from scripts.benchmark_model_speed import benchmark_model_speed_parser
from scripts.dataset_info import dataset_info_parser
from scripts.model_test import model_test_parser
from scripts.model_train import model_train_parser
from scripts.network_info import network_info_parser
from scripts.preprocess_data import preprocess_data_parser
from scripts.rank_analysis_dir import rank_analysis_dir_parser
from scripts.sim_data import sim_data_parser

script_parsers = [
    batch_model_test_parser,
    batch_model_train_parser,
    benchmark_model_speed_parser,
    dataset_info_parser,
    model_test_parser,
    model_train_parser,
    network_info_parser,
    preprocess_data_parser,
    rank_analysis_dir_parser,
    sim_data_parser,
]
