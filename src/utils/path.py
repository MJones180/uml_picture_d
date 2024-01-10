from pathlib import Path
import shutil


def copy_files(from_path, to_path):
    shutil.copy(from_path, to_path)


def delete_file(path, quiet=False):
    try:
        Path(path).unlink(True)
        if not quiet:
            print('Delete successful.')
    except FileNotFoundError:
        if not quiet:
            print('Nothing to delete.')


def delete_dir(path, quiet=False):
    try:
        shutil.rmtree(path)
        if not quiet:
            print('Delete successful.')
    except FileNotFoundError:
        if not quiet:
            print('Nothing to delete.')


def make_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def path_exists(path):
    return Path(path).exists()


def path_file(path):
    return Path(path).name


def path_parent(path):
    return Path(path).parent
