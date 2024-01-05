import sys


def terminate_with_message(message):
    """Terminate the current process and display a message to the user.

    Parameters
    ----------
    message : str
        Message to write out to the CLI.

    Notes
    -----
    This function will terminate the process via `sys.exit()`.
    """

    print()
    print('-' * 22)
    print('- TERMINATING SCRIPT -')
    print('-' * 22)
    print()
    print('REASON:')
    print(f'  {message}')
    print()
    print('END.')
    sys.exit()
