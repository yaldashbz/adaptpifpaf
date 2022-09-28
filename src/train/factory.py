import argparse


def cli(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('generic train parameters')

    group.add_argument('--device-name', default='cuda',
                       help='CPU or CUDA')
    group.add_argument('--dp', default=False, action='store_true',
                       help='[experimental] DataParallel')
    group.add_argument('--lr', '--learning-rate', default=2.5e-4, type=float,
                       metavar='LR', help='initial learning rate')
    parser.add_argument('--power', default=0.9, type=float, help='power for learning rate decay')
    # parser.add_argument('--momentum', default=0, type=float, metavar='M',
    #                     help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR is multiplied by gamma on schedule.')
    group.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                       help='evaluate model on validation set')
    group.add_argument('--epochs', default=80, type=int, metavar='N',
                       help='number of total epochs to run')
    group.add_argument('--max_epoch', default=100, type=int, metavar='N',
                       help='number of total epochs to run')
    group.add_argument('--start-epoch', default=0, type=int, metavar='N',
                       help='manual epoch number (useful on restarts)')
    group.add_argument('--resume', default='', type=str, metavar='PATH',
                       help='path to latest checkpoint (default: none)')
