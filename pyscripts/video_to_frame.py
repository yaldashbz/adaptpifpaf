import os
import sys

"""
Notes:
    - VID001 & VID009 data is not valid. some frames has missed values. frames are available in:
        /data/yalda-data/RBDVideo/frames
        /data/yalda-data/AdaptPifPaf/test
        - example: frame_100_VID001_11.24 - 11.25_CH01-2021-11-25-03-44-52_03-49-40.png
        - example: frame_44_VID009_2.21-2.22_CH01-2022-02-21-22-34-37_22-40-52.png
    - VID003/12.2-12.3/CH08-2021-12-03-05-53-47_05-53-47 what is this? no frames?!
"""


def cli():
    import argparse

    base = '../rbd_videos/home_videos'
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--start', type=int)
    parser.add_argument('--end', type=int)
    parser.add_argument('--frame-count', type=int)

    parser.add_argument('--root', default=base)
    parser.add_argument('--save-dir', default='../datasets/rbd_dataset')
    args = parser.parse_args()
    return args


def main():
    args = cli()

    convertor_kwargs = {'start': args.start, 'end': args.end, 'frame_count': args.frame_count}
    convert_all_data(args.root, args.save_dir, **convertor_kwargs)


if __name__ == '__main__':
    sys.path.append(os.getcwd())

    from src.datasets.video_to_frame import convert_all_data

    main()
    # vid_id = 'VID007'
    # date = '1.23 -1.24'
    # filename = 'CH01-2022-01-23-23-03-55_23-23-46.avi'
    # convertor = VideoToFrame(
    #     frame_count=15
    # )
    #
    # frames, fps = convertor.extract_frames(f'../rbd_videos/home_videos/{vid_id}/{date}/{filename}',
    #                                        save_dir=f'../datasets/rbd_dataset/{vid_id}/{date}/{filename.split(".")[0]}')
    # print(np.asarray(frames).shape, fps)
