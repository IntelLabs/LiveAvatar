
import os
import glob
import shutil

import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='pims')

import os
import queue
import time
import torch
import torch.multiprocessing as mp

from utils import log
from datasets.celebvhq import load_data
from tracking.face_tracker import OfflineFaceTracker


def run_singleprocessing_track(args):
    device = f"cuda" if args.num_gpus > 0 and torch.cuda.is_available() else "cpu"
    face_tracker = OfflineFaceTracker(tasks=args.tasks,
                                      asset_dir=args.asset_dir,
                                      batch_size=args.batchsize,
                                      progress=args.progress,
                                      fov=args.fov,
                                      device=device)
    video_paths = sorted(os.listdir(args.video_dir))

    for idx in range(args.st, args.nd):

        clip_name = video_paths[idx]
        video_path = os.path.join(args.video_dir, clip_name)

        print(f"==== Video {idx}: {clip_name} ====")

        face_tracker.track_video(
            os.path.join(video_path, 'images'),
            output_dir=args.output_dir,
            output_sub_folder=clip_name,
            clip_name=clip_name,
            clip_id=idx,
            interval=args.frame_interval,
            max_frame_count=args.max_frame_count,
            max_clip_length=args.max_clip_length,
            overwrite=args.overwrite,
            show=args.show,
            wait=args.wait
        )

def copy_videos():
    ROOT_DIR  = "../../../data/datasets/nersemble"
    participant_ids = ['017', '018', '024', '030']
    camera = '222200037'
    sequence = 'EXP-1-head'

    outdir = os.path.join(ROOT_DIR, 'test')
    os.makedirs(outdir, exist_ok=True)

    for id in participant_ids:
        infile = os.path.join(ROOT_DIR, 'data', id, 'sequences', sequence, 'images', 'cam_'+camera+'.mp4')
        outfile = os.path.join(outdir, f"{id}_{sequence}_{camera}.mp4")
        print(infile, '>>>',  outfile)
        shutil.copyfile(infile, outfile)


if __name__ == '__main__':
    import json
    import argparse

    def bool_str(x):
        return str(x).lower() in ['True', 'true', '1']

    file_dir = os.path.dirname(os.path.realpath(__file__))
    default_asset_dir = os.path.join(file_dir, "../assets")
    default_data_dir = os.path.join(file_dir, "../../../data/datasets/nersemble/test")

    parser = argparse.ArgumentParser()

    parser.add_argument('--root', type=str, default=default_data_dir)
    parser.add_argument('--asset_dir', type=str, default=default_asset_dir)
    parser.add_argument('--video_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)

    parser.add_argument('--st', type=int, default=0)
    parser.add_argument('--nd', type=int, default=1)
    parser.add_argument('--overwrite', type=bool_str, default=True)

    parser.add_argument('-g', '--num_gpus', type=int, default=1)
    parser.add_argument('-p', '--num_processes', type=int, default=1)
    parser.add_argument('-b', '--batchsize', default=8, type=int, metavar='N')

    parser.add_argument('--fov', type=float, default=30.0)
    parser.add_argument('--frame_interval', type=int, default=None)
    parser.add_argument('--max_frame_count', type=int, default=None)
    parser.add_argument('--max_clip_length', type=int, default=None)

    parser.add_argument('--show', type=bool_str, default=False)
    parser.add_argument('--wait', type=bool_str, default=False)

    # parser.add_argument('-t', '--tasks',  nargs='+', default=['segment', 'detect', 'align', 'metadata'], metavar='N')
    parser.add_argument('-t', '--tasks',  nargs='+', default=['segment'], metavar='N')

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.root, 'processed_vfhq')
    os.makedirs(args.output_dir, exist_ok=True)

    if args.video_dir is None:
        args.video_dir = os.path.join(args.root, 'GAGAvatar_track')

    # json_path = os.path.join(args.root, 'celebvhq_info.json')

    # with open(json_path) as f:
    #     total_vid_count = len(json.load(f)['clips'])
    total_vid_count = len(os.listdir(args.video_dir))

    if args.st is None:
        args.st = 0

    if args.nd is None:
        args.nd = total_vid_count

    args.nd = min(total_vid_count, args.nd)

    if args.st < 0:
        args.st = total_vid_count + args.st

    if args.nd < 0:
        args.nd = total_vid_count + args.nd

    num_videos = min(args.nd - args.st, total_vid_count)
    num_workers = max(1, args.num_gpus) * args.num_processes

    log.info(args)
    log.info(f"Number of videos in dataset: {total_vid_count}")
    log.info(f"Number of videos to process: {num_videos}")
    log.info(f"Number of GPUs: {args.num_gpus}")
    log.info(f"Number of processes per device: {args.num_processes}")
    log.info(f"Number of workers: {num_workers}")
    log.info(f"Number of videos per worker: {num_videos // num_workers}")

    t = time.time()

    args.progress = True
    run_singleprocessing_track(args)

    log.info(f"All videos completed. Time={time.time()-t:.2f}s")






