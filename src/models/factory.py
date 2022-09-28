import argparse

from src.models.openpifpaf_refine import openpifpaf_refine
from src.models.respose_refine import pose_resnet_refine_mt_multida

# urls for pose estimation module
CHECKPOINT_URLS = dict()

MODELS = dict(
    openpifpaf_refine=openpifpaf_refine,
    respose_refine=pose_resnet_refine_mt_multida
)


class Factory:
    checkpoint = None
    network = None
    resnet_layers = 50
    pretrained_path = None
    dual_branch = False

    def factory(self, n_joints):
        return MODELS[self.network](
            num_classes=n_joints,
            resnet_layers=self.resnet_layers,
            pretrained_path=self.pretrained_path,
            dual_branch=self.dual_branch
        )

    @classmethod
    def configure(cls, args: argparse.Namespace):
        cls.checkpoint = args.checkpoint
        cls.network = args.network
        cls.resnet_layers = args.resnet_layers
        cls.pretrained_path = args.pretrained_path
        cls.dual_branch = args.dual_branch

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('network configuration')
        group.add_argument(
            '--network', default=cls.network,
            help=(
                'Network Architecture for student and teacher'
            )
        )
        group.add_argument('--dual-branch', action='store_true',
                           help='Whether has two branches in refinenet')
        # group.add_argument('--logit-distance-cost', default=0.01, type=float, metavar='WEIGHT')
        group.add_argument('--ema-decay', default=0.999, type=float, metavar='ALPHA',
                           help='ema variable decay rate (default: 0.999)')
        group.add_argument('--consistency', default=90.0, type=float, metavar='WEIGHT',
                           help='use consistency loss with given weight (default: None)')
        group.add_argument('--consistency-rampup', default=10, type=int, metavar='EPOCHS',
                           help='length of the consistency loss ramp-up')
        group.add_argument('--occlusion-aug', action='store_true', help='whether add occlusion augment')
        group.add_argument('--num-occluder', type=int, default=8, help='number of occluder to add in')
        group.add_argument(
            '--pretrained-path', default=cls.pretrained_path,
            type=str, metavar='PATH',
            help='path to latest checkpoint (default: none)'
        )
        group.add_argument(
            '--resnet-layers', default=cls.resnet_layers,
            type=int, metavar='N',
            help='Number of resnet layers',
            choices=[18, 34, 50, 101, 152]
        )
        available_checkpoints = [n for n, url in CHECKPOINT_URLS.items()]
        group.add_argument(
            '--checkpoint', default=cls.checkpoint,
            help=(
                'Path to a local checkpoint for pose estimation module. '
                'Or provide one of the following to download a pretrained model: {}'
                ''.format(available_checkpoints)
            )
        )
