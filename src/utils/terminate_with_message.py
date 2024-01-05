import sys


def terminate_with_message(message):
    print(f'''
{'-' * 18}
TERMINATING SCRIPT

REASON:
  {message}
{'-' * 18}
''')
    sys.exit()
