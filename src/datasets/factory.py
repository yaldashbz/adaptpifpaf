from src.datasets import (
    synthetic_animal_sp_all,
    real_animal_crop_all,
    real_animal_all
)

DATASETS = dict(
    real_animal_all=real_animal_all,
    real_animal_crop_all=real_animal_crop_all,
    synthetic_animal_sp_all=synthetic_animal_sp_all
)


def factory(dataset: str, **kwargs):
    if dataset not in DATASETS.keys():
        raise Exception(f'dataset {dataset} unknown')
    return DATASETS[dataset](**kwargs)


def get_joints_num(dataset: str):
    return 18
    # if dataset not in DATASETS.keys():
    #     raise Exception(f'dataset {dataset} unknown')
    # try:
    #     return DATASETS[dataset].n_joints
    # except ValueError:
    #     raise Exception(f'dataset {dataset} does not have "n_joints"')


def cli(parser):
    group = parser.add_argument_group('generic data module parameters')
    # source dataset
    group.add_argument('--src-dataset')

    # target dataset
    group.add_argument('--trg-dataset')
    group.add_argument('--trg-dataset-crop')

    # batch sizes
    group.add_argument('--train-batch',
                       default=8, type=int,
                       help='train batch size')
    group.add_argument('--test-batch',
                       default=8, type=int,
                       help='test batch size')

    # workers
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    # animal dataset
    parser.add_argument('--image-path', default='./animal_data/', type=str,
                        help='path to images')
    group.add_argument('--animal', default='all', type=str,
                       help='horse | tiger | sheep | hound | elephant')
    group.add_argument('--year', default=2014, type=int, metavar='N',
                       help='year of coco dataset: 2014 (default) | 2017)')
    group.add_argument('--inp-res', default=256, type=int,
                       help='input resolution (default: 256)')
    group.add_argument('--out-res', default=64, type=int,
                       help='output resolution (default: 64, to gen GT)')

    # data processing
    parser.add_argument('-f', '--flip', dest='flip', action='store_true',
                        help='flip the input during validation')
    parser.add_argument('--sigma', type=float, default=1,
                        help='Groundtruth Gaussian sigma.')
    parser.add_argument('--scale-factor', type=float, default=0.4,
                        help='Scale factor (data aug).')
    parser.add_argument('--rot-factor', type=float, default=45,
                        help='Rotation factor (data aug).')
    parser.add_argument('--sigma-decay', type=float, default=0,
                        help='Sigma decay rate for each epoch.')
    parser.add_argument('--label-type', metavar='LABELTYPE', default='Gaussian',
                        choices=['Gaussian', 'Cauchy'],
                        help='Labelmap dist type: (default=Gaussian)')
    parser.add_argument('--percentage', type=float, default=0.6,
                        help='Percentage of data to be filtered out.')
    parser.add_argument('--stage', type=str, default='1', help='which stage to load psudo label ')
    parser.add_argument('--train_on_all_cat', action='store_true', help='whether train on all categories')
    parser.add_argument('--tent', action='store_true', help='model adaptation with tent', default=False)
    parser.add_argument('--generate_pseudol', action='store_true', help='whether generate pseudo labels')

    # online reliable points mining
    parser.add_argument('--min-kpts', type=int, default=9, help='number of minimum hard keypoints')
    parser.add_argument('--start-mine-epoch', type=int, default=2, help='start epoch to mine ')
    parser.add_argument('--reduce-interval', type=int, default=1, help='start epoch to mine ')

    # initial pseudo labels
    parser.add_argument("--gamma_", type=float, default=15.0, help="target dataset loss coefficient")
    parser.add_argument('--gamma_rampdown', type=int, default=15)
    parser.add_argument('--min_gamma', type=float, default=8)