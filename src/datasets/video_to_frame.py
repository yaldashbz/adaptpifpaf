import multiprocessing as mp
import os
import shutil
from itertools import repeat

import cv2
import numpy as np


class VideoToFrame:
    def __init__(
            self,
            start: int = None,
            end: int = None,
            frame_count: int = None,
            num_processes: int = 3
    ):
        self.start = start
        self.end = end
        self.frame_count = frame_count
        self.num_processes = num_processes

    @classmethod
    def _get_save_path(cls, group_number, save_dir):
        return os.path.join(save_dir, str(group_number)) + '.npy'

    @classmethod
    def load_frames(cls, data_dir):
        frames = list()
        for np_file in os.listdir(data_dir):
            frames.extend(np.load(os.path.join(data_dir, np_file)))
        return frames

    def _merge_frames(self, save_dir):
        frames = list()
        try:
            frames = self.load_frames(save_dir)
            shutil.rmtree(save_dir)
            np.save(save_dir + '.npy', np.array(frames))
        except Exception as e:
            print(e.args[0])
        return frames

    def extract_frames(self, video_path: str, save_dir: str):
        num_processes = self.num_processes
        start = self.start or 0
        end = self.end if self.end else float('inf')
        assert start <= end
        assert os.path.isfile(video_path)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        pool = mp.Pool(num_processes)
        pool.starmap(
            process_video,
            zip(
                range(num_processes),
                repeat(video_path),
                repeat(save_dir),
                repeat(start),
                repeat(end),
                repeat(self.frame_count)
            )
        )

        return self._merge_frames(save_dir), fps


def process_video(
        group_number: int,
        video_path: str,
        save_dir: str,
        start: int,
        end: int,
        frame_count: int
):
    cap = cv2.VideoCapture(video_path)
    num_processes = mp.cpu_count()
    frame_jump_unit = cap.get(cv2.CAP_PROP_FRAME_COUNT) // num_processes
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_jump_unit * group_number)
    proc_frames = 0

    frames = list()
    while proc_frames < frame_jump_unit:
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)
        proc_frames += 1

    end = min(end, len(frames))
    if end == 0:
        return
    indices = np.random.choice(list(range(start, end)), frame_count)
    frames = np.asarray(frames)[indices.astype(int)]
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, str(group_number)) + '.npy', np.array(frames))
    cap.release()


def convert_all_data(base_dir, save_dir, **convertor_kwargs):
    convertor = VideoToFrame(**convertor_kwargs)
    for category_id in os.listdir(base_dir):
        category_path = os.path.join(base_dir, category_id)
        for date in os.listdir(category_path):
            date_path = os.path.join(category_path, date)
            for vid_file in os.listdir(date_path):
                vid_path = os.path.join(date_path, vid_file)
                np_path = os.path.join(
                    save_dir, category_id, date, vid_file.split('.')[0])
                if os.path.isfile(np_path + '.npy'):
                    continue
                print(f'extracting for {vid_file} with ID {category_id} and date {date}')
                frames, fps = convertor.extract_frames(vid_path, np_path)
                print('frames shape: ', np.array(frames).shape)
                print('fps: ', fps)
