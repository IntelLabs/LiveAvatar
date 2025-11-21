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


class Worker(mp.Process):
    def __init__(self, job_queue: mp.Queue, device, cfg):
        super().__init__()
        self.job_queue = job_queue
        self.device = device
        self.cfg = cfg

    def run(self):
        raise NotImplementedError


class TrackWorker(Worker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        if self.device != "cpu":
            torch.cuda.device(self.device)

        face_tracker = OfflineFaceTracker(
            tasks=self.cfg.tasks,
            asset_dir=self.cfg.asset_dir,
            batch_size=self.cfg.batchsize,
            fov=self.cfg.fov,
            progress=self.cfg.progress,
            device=self.device
        )

        while not self.job_queue.empty():
            try:
                job_data = self.job_queue.get_nowait()
                idx, clip_name = job_data

                log.info(f"==== Video {idx}: {clip_name} (Device {self.device}, {torch.cuda.current_device()}) ====")

                # clip_name = os.path.splitext(save_vid_name)[0]
                video_path = os.path.join(self.cfg.video_dir, clip_name)

                face_tracker.track_video(
                    video_path,
                    output_dir=self.cfg.output_dir,
                    output_sub_folder=clip_name,
                    clip_name=clip_name,
                    clip_id=idx,
                    interval=self.cfg.frame_interval,
                    max_frame_count=self.cfg.max_frame_count,
                    max_clip_length=self.cfg.max_clip_length,
                    overwrite=self.cfg.overwrite,
                    show=self.cfg.show,
                    wait=self.cfg.wait
                )

            except queue.Empty:
                break


def run_multiprocessing(args):
    mp.set_start_method("spawn", force=True)

    video_paths = os.listdir(args.video_dir)
    job_queue = mp.Queue()
    # for job_data in load_data(json_path, args.st, args.nd):
    #     job_queue.put(job_data)
    for idx in range(len(video_paths))[args.st: args.nd]:
        job_queue.put((idx, video_paths[idx]))

    worker = TrackWorker

    def get_device(worker_id, num_gpus):
        device = "cpu"
        if num_gpus > 0 and torch.cuda.is_available():
            device = f"cuda:{worker_id % args.num_gpus}"
        return device

    num_workers = max(1, args.num_gpus) * args.num_processes
    worker_devices = [get_device(i, args.num_gpus) for i in range(num_workers)]
    print("Worker devices:")
    print(worker_devices)
    workers: list[Worker] = [worker(job_queue, device, args) for device in worker_devices]

    for worker in workers:
        worker.start()

    for worker in workers:
        worker.join()


def run_singleprocessing_track(args):
    device = f"cuda" if args.num_gpus > 0 and torch.cuda.is_available() else "cpu"
    face_tracker = OfflineFaceTracker(tasks=args.tasks,
                                      asset_dir=args.asset_dir,
                                      batch_size=args.batchsize,
                                      progress=args.progress,
                                      fov=args.fov,
                                      device=device)
    video_paths = os.listdir(args.video_dir)

    for idx in range(args.st, args.nd):

        clip_name = video_paths[idx]
        video_path = os.path.join(args.video_dir, clip_name)

        print(f"==== Video {idx}: {clip_name} ====")

        face_tracker.track_video(
            video_path,
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


if __name__ == '__main__':
    import json
    import argparse

    def bool_str(x):
        return str(x).lower() in ['True', 'true', '1']

    split = "test"
    file_dir = os.path.dirname(os.path.realpath(__file__))
    default_asset_dir = os.path.join(file_dir, "../assets")
    default_data_dir = os.path.join(file_dir, "../../../data/datasets/VFHQ", split)

    parser = argparse.ArgumentParser()

    parser.add_argument('--root', type=str, default=default_data_dir)
    parser.add_argument('--asset_dir', type=str, default=default_asset_dir)
    parser.add_argument('--video_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)

    # 7, 13, 24, 54
    parser.add_argument('--st', type=int, default=0)
    parser.add_argument('--nd', type=int, default=4)
    parser.add_argument('--overwrite', type=bool_str, default=True)

    parser.add_argument('-g', '--num_gpus', type=int, default=1)
    parser.add_argument('-p', '--num_processes', type=int, default=1)
    parser.add_argument('-b', '--batchsize', default=8, type=int, metavar='N')

    parser.add_argument('--fov', type=float, default=30.0)
    parser.add_argument('--frame_interval', type=int, default=None)
    parser.add_argument('--max_frame_count', type=int, default=24)
    parser.add_argument('--max_clip_length', type=int, default=None)

    parser.add_argument('--show', type=bool_str, default=False)
    parser.add_argument('--wait', type=bool_str, default=False)

    parser.add_argument('-t', '--tasks',  nargs='+', default=['segment', 'detect', 'align', 'metadata'], metavar='N')

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.root, 'processed_vfhq')
    os.makedirs(args.output_dir, exist_ok=True)

    if args.video_dir is None:
        if split == "train":
            args.video_dir = os.path.join(args.root, 'mp4')
        else:
            args.video_dir = os.path.join(args.root, 'GT/Interval1_512x512_LANCZOS4')

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

    if num_workers > 1:
        args.progress = False
        run_multiprocessing(args)
    else:
        args.progress = True
        run_singleprocessing_track(args)

    log.info(f"All videos completed. Time={time.time()-t:.2f}s")
