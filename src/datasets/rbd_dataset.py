import random

import numpy as np
import torch
import tqdm

from datasets.utils import get_samples


class RBDIterableDataset(torch.utils.data.IterableDataset):
    def __init__(
            self,
            root: str,
            is_train: bool,
            epoch_size: int = None,
            frames_count: int = None,
            start: int = None,
            end: int = None,
            frame_transform=None,
            video_transform=None,
    ):
        if start is not None and end:
            assert start <= end
        super(RBDIterableDataset).__init__()

        self.samples = get_samples(root)

        # allow for temporal jittering
        if epoch_size is None:
            epoch_size = len(self.samples)
        self.epoch_size = epoch_size
        self.frames_count = frames_count
        self.frame_transform = frame_transform
        self.video_transform = video_transform
        self.start = start or 0
        self.end = end

    def __iter__(self):
        for _ in tqdm.tqdm(range(self.epoch_size)):
            # get random sample
            path, target = random.choice(self.samples)
            video = np.load(path)
            print(len(video))
            if not self.end:
                self.end = len(video)
            video = torch.from_numpy(video[self.start:self.end])
            if self.frames_count:
                video = torch.randint(len(video), (self.frames_count,))
            if self.video_transform:
                video = self.video_transform(video)
            output = {
                'path': path,
                'video': video,
                'target': target,
                'start': self.start,
                'end': self.start + self.clip_len
            }
            print('end of this')
            yield output
