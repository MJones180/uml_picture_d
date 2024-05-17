import add_packages_dir_to_path  # noqa: F401

import argparse
from script_parsers import script_parsers
from sys import argv


def main(raw_args=None):
    """
    Setup the argparser, parse the arguments, and call the correct script.

    Parameters
    ----------
    raw_args : list, optional
        Arguments to pass to the subparser.
    """
    # Set the default for the `raw_args`
    if raw_args is None:
        raw_args = argv[1:]
    # Create an argparser object
    cli_parser = argparse.ArgumentParser()
    # Allow for subparsers to be created
    subparsers = cli_parser.add_subparsers()
    # Add the subparser from each script
    for subparser_creator in script_parsers:
        subparser_creator(subparsers)
    # Parse the `raw_args`, will automatically call the correct subparser
    matched_args, unmatched_args = cli_parser.parse_known_args(raw_args)
    # Convert to a dictionary so values can be accessed
    cli_args_dict = vars(matched_args)
    # Add all CLI overrides to the matched_args to make them accessible
    cli_args_dict['unmatched_args'] = unmatched_args
    # Ensure a subparser was called
    if 'main' not in cli_args_dict:
        cli_parser.error('No command provided')
    # Pass all argparser arguments to the popped off `main` object
    cli_args_dict.pop('main')(cli_args_dict)


if __name__ == '__main__':
    main()
