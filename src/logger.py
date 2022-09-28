import logging
import argparse

LOG = logging.getLogger(__name__)


def cli(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('logger')
    group.add_argument('-q', '--quiet', default=False, action='store_true',
                       help='only show warning messages or above')
    group.add_argument('--debug', default=False, action='store_true',
                       help='print debug messages')
    group.add_argument('--log-stats', default=False, action='store_true',
                       help='enable stats logging')