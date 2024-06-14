"""
Remove old models from the `tag_lookup.json` file.
"""

from glob import glob
from utils.constants import OUTPUT_P, TAG_LOOKUP_F, TRAINED_MODELS_P
from utils.json import json_load, json_write
from utils.path import path_exists
from utils.printing_and_logging import step_ri, title


def prune_tag_lookup_parser(subparsers):
    """
    Example commands:
        python3 main.py prune_tag_lookup
    """
    subparser = subparsers.add_parser(
        'prune_tag_lookup',
        help='prune the `tag_lookup.json` file',
    )
    subparser.set_defaults(main=prune_tag_lookup)


def prune_tag_lookup(cli_args):
    title('Prune tag lookup script')

    step_ri('Finding current model tags')
    # Tags found within the `trained_models` folder
    existing_model_tags = [
        path.split('/')[-1] for path in glob(f'{TRAINED_MODELS_P}/*')
    ]
    print(', '.join(existing_model_tags))

    step_ri('Pruning old tags')
    tag_lookup_path = f'{OUTPUT_P}/{TAG_LOOKUP_F}'
    if path_exists(tag_lookup_path):
        tag_lookup = json_load(tag_lookup_path)
    else:
        tag_lookup = {}
    # Create a tag lookup with only the existing keys
    updated_tag_lookup = {
        key: val
        for key, val in tag_lookup.items() if key in existing_model_tags
    }
    removed_tags = ', '.join(tag_lookup.keys() - updated_tag_lookup.keys())
    print(f'Removing the following tags: {removed_tags}')
    print('Writing out updated `tag_lookup.json`')
    json_write(tag_lookup_path, updated_tag_lookup)
