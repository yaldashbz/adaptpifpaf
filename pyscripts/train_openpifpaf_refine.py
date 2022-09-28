def cli():
    import argparse

    base = '../datasets/rbd_dataset'
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--target-dir', default=base)
    parser.add_argument('--source-dir', default=base)

    args = parser.parse_args()
    return args


def main():
    args = cli()


if __name__ == '__main__':
    main()
