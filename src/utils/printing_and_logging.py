import builtins

# Must keep a reference to the default printer
base_print = builtins.print

# Indentation block
INDENT_BLOCK = ' ' * 4

# Current indentation level
_current_indent_level = 0


def _modified_printer(*args, **kwargs):
    return base_print(
        INDENT_BLOCK * _current_indent_level,
        *args,
        **kwargs,
    )


# Override the default printer to take into account indentation level
builtins.print = _modified_printer


def inc_print_indent():
    """
    Increase the global indentation printing level by one.
    """
    global _current_indent_level
    _current_indent_level += 1


def dec_print_indent():
    """
    Decrease the global indentation printing level by one.
    """
    global _current_indent_level
    if _current_indent_level > 0:
        _current_indent_level -= 1


def reset_print_indent():
    """
    Reset the global indentation printing level.
    """
    global _current_indent_level
    _current_indent_level = 0


def divider(length=50):
    """
    Divider made with the `-` character surrounded by new lines.
    """
    print('')
    print('-' * length)
    print('')


def title(text):
    """
    Print a title statement.
    """
    print(text.upper())
    print('=' * len(text))


def step(text):
    """
    Print a step statement and increase indentation by one.
    """
    print('')
    print(f'{text}...')
    inc_print_indent()


def step_ri(text):
    """
    Call the `step` function and reset previous indentation.
    """
    reset_print_indent()
    step(text)
