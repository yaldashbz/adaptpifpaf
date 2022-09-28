import argparse
import os
import sys
import wandb


def _get_parser():
    parser = argparse.ArgumentParser(
        prog='python3 -m adaptpifpaf.train',
        usage='%(prog)s [options]',
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    return parser


def _configure(args):
    models.Factory.configure(args)
    # show.configure(args)


def cli():
    parser = _get_parser()
    logger.cli(parser)
    models.Factory.cli(parser)
    datasets.cli(parser)
    train.cli(parser)
    # show.cli(parser)
    args = parser.parse_args()
    _configure(args)
    return args


def main():
    args = cli()
    trainer = train.MeanTeacherTrainer(args)
    trainer.train()


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    from src import logger, models, train, datasets

    wandb.init(project="UDA-Animal-Pose", entity="yaldashbz", name='train_new_project')
    main()
