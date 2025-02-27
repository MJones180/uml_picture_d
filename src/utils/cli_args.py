from utils.constants import ARGS_F
from utils.json import json_load, json_write


def load_cli_args(path):
    return json_load(f'{path}/{ARGS_F}')


def save_cli_args(path, cli_args, script_name):
    cli_args['_script_name'] = script_name
    json_write(f'{path}/{ARGS_F}', cli_args)
