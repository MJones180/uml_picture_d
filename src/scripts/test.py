def test_parser(subparsers):
    subparser = subparsers.add_parser(
        'test',
        help='Desc',
    )
    subparser.set_defaults(main=test)


def test(cli_args):
    print('Hi')
