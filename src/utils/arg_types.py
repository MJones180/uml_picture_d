import argparse
from os.path import isdir, isfile, normpath


def dir_path(path):
    if isdir(path):
        return normpath(path)
    raise argparse.ArgumentTypeError(f'{path} is not a valid directory.')


def file_path(path):
    if isfile(path):
        return path
    raise argparse.ArgumentTypeError(f'{path} is not a valid file.')
