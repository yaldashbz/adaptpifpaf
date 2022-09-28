import os

from torchvision.datasets.folder import make_dataset


def find_classes(base_dir):
    """
    :param base_dir: base directory of home videos
    :return: patients id as classes and the class_to_idx doct
    """
    classes = [d.name for d in os.scandir(base_dir) if d.is_dir()]
    classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def get_samples(base_dir: str, extensions=('.mp4', '.avi')):
    _, class_to_idx = find_classes(base_dir)
    return make_dataset(base_dir, class_to_idx, extensions=extensions)
