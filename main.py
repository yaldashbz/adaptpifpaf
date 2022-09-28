import os

import matplotlib.pyplot
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms as t

from datasets.rbd_dataset import RBDIterableDataset

if __name__ == '__main__':
    vid_id = 'VID009'
    date = '2.21-2.22'
    name = 'CH01-2022-02-21-22-34-37_22-40-52'
    idx = 44
    frames = np.load(f'../datasets/rbd_dataset/{vid_id}/{date}/{name}.npy')
    print(frames.shape)
    # plt.figure()
    # print(frames[0])
    plt.imshow(frames[idx])
    # # os.mkdir('./test')
    plt.show()
    plt.imsave(f'./test/frame_{idx}_{vid_id}_{date}_{name}.png', frames[idx])
    print(f'frame_{idx}_{vid_id}_{date}_{name}.png')
    # matplotlib.pyplot.show()
    # sys.path.append(os.getcwd())

    # p = '../rbd_videos/home_videos'
    # transforms = [t.Resize((112, 112))]
    # frame_transform = t.Compose(transforms)
    #
    # dataset = RBDIterableDataset(p, epoch_size=5, frame_transform=frame_transform, frames_count=16)
    #
    # from torch.utils.data import DataLoader
    #
    # loader = DataLoader(dataset, batch_size=12)
    # data = {"video": [], 'start': [], 'end': [], 'tensorsize': []}
    # for batch in loader:
    #     for i in range(len(batch['path'])):
    #         data['video'].append(batch['path'][i])
    #         data['start'].append(batch['start'][i].item())
    #         data['end'].append(batch['end'][i].item())
    #         data['tensorsize'].append(batch['video'][i].size())
    # print(data)
