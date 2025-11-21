import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='pims')
import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

from typing import Literal
import tqdm
import cv2
import numpy as np
import multiprocessing as mp
import time
import tyro
from dataclasses import dataclass

from models.live_avatar import LiveAvatar
from configs.config import Config, ModelConfig
from tracking.stream_face_tracker import StreamFaceTracker
from utils.util import create_camera, pad_to_square, get_images, transform_camera2, orbit_camera
from visualization.vis import show_image, to_image
from server.util import LastOnlyQueue
from server.webcam import WebcamGrabber
from server.socket_io import ImageSocketReceiver, ImageSocketSender
# from server import webrtc


def paste_image(img_large, img_small, top, left):
    h, w = img_small.shape[:2]

    bottom = min(top + h, img_large.shape[0])
    right = min(left + w, img_large.shape[1])
    paste_h = bottom - top
    paste_w = right - left

    if paste_h > 0 and paste_w > 0:
        img_large[top:bottom, left:right] = img_small[:paste_h, :paste_w]

    return img_large


class TrackWorker(mp.Process):

    def __init__(
            self,
            frame_queue: LastOnlyQueue,
            output_queue: LastOnlyQueue,
            detect_face: bool = True,
            show: bool = True,
            config: dict | None = None
    ):
        super().__init__()
        self.frame_queue = frame_queue
        self.result_queue = output_queue
        self.last_timestamp = time.time()
        self.show = show
        self.detect_face = detect_face
        self.config = config

    def run(self):
        tracker = StreamFaceTracker(
            image_size=512,
            asset_dir="./assets",
            reset_interval_sec=None,
            update_freq=1,
            gpu=False,
            min_confidence=0.5 if self.detect_face else 0
        )

        while True:
            try:
                full_image, frame_recv_time = self.frame_queue.get()
            except:
                continue

            assert isinstance(full_image, np.ndarray)

            if self.show:
                show_image("[Tracker] Input frame", full_image)
                cv2.waitKey(1)

            zoom_factor = self.config['camera_distance']

            new_distance = tracker.camera_distance * zoom_factor
            new_pos = np.array([[0.5, -0.5, new_distance]])
            tracker._canonical_face_alignment.update_cam_pos(new_pos)

            tracking_result = tracker.track(full_image, int(frame_recv_time*1000), crop_input=self.detect_face, show=False)

            update_duration = time.time() - self.last_timestamp
            self.last_timestamp = time.time()
            fps = 1 / update_duration
            latency = time.time() - frame_recv_time

            if tracking_result is not None:

                self.result_queue.put((full_image, tracking_result, frame_recv_time))

                if self.show and False:
                    disp_result = tracking_result['crop'].copy()
                    disp_result = cv2.putText(disp_result, f"fps={int(fps)}", (2, 18),
                                             cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255),
                                             1, cv2.LINE_AA)
                    disp_result = cv2.putText(disp_result, f"lat={int(latency*1000)}ms", (2, 50),
                                              cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255),
                                              1, cv2.LINE_AA)
                    show_image("[Tracker] Detected face", disp_result)
                    cv2.waitKey(1)


class Worker(mp.Process):

    def __init__(
            self,
            input_queue: LastOnlyQueue,
            output_queue: LastOnlyQueue,
            display_queue: LastOnlyQueue,
            pc_queue: LastOnlyQueue,
            control_queue: mp.Queue,
            checkpoint: str,
            config: dict,
            device: str = "cuda",
            show: bool = True
    ):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.display_queue = display_queue
        self.pc_queue = pc_queue
        self.control_queue = control_queue
        self.device = device
        self.checkpoint = checkpoint
        self.show = show
        self.camera_id = None
        self.config = config
        azimuths, elevations = np.meshgrid([-45, 0, 45], [-45, 0, 45])
        self.fixed_cameras = [
            create_camera(azi, ele, distance=2.5, fov=30).to(device) for azi, ele in zip(azimuths.ravel(), elevations.ravel())
        ]

    def run(self):

        @dataclass
        class Model(ModelConfig):
            image_size: int = self.config['image_size']
            levels_dec: int = 4
            num_patches: int = 24
            azimuth_range: float = 180.0
            backbone: str = 'vits'
            blocks_dec: int = 2
            dim_expr: int = 64

        @dataclass
        class AvatarConfig(Config):
            net: Model
            render_size: int = self.config['render_size']
            device: str = self.device
            background_color: tuple[float, float, float] = (0.1,)*3

        avatar = LiveAvatar(
            checkpoint=self.checkpoint,
            pred_expr=self.config['pred_expr'],
            cfg=AvatarConfig(Model()),
        )

        ref_img = None
        W, H = self.config['image_size'], self.config['image_size']
        t0 = None

        while True:
            try:
                full_frame, tracking_result, timestamp = self.input_queue.get()
            except:
                continue

            if t0 is None:
                t0 = timestamp

            try:
                cmd, timestamp = self.control_queue.get_nowait()
            except:
                cmd = {}

            if 'set_ref_img' in cmd:
                print(f"Updating identity...")
                # avatar.reset()

                ref_img = cmd['set_ref_img']
                if ref_img is not None:

                    # avatar.set_identity_and_expression(ref_img)
                    avatar.add_identity(ref_img)
                    if len(avatar.embeddings) > 2:
                        avatar.remove_identity(0)

            tracked_image = tracking_result['crop']
            tracked_keypoints = tracking_result['keypoints_crop'][..., :2]

            if ref_img is None:
                # Same frame reconstruction
                avatar.set_identity_and_expression(tracked_image, keypoints=tracked_keypoints)
            else:
                # Cross-identity
                avatar.set_expression(tracked_image, keypoints=tracked_keypoints)

            avatar.update()

            camera_id = self.config['camera_id']
            camera = tracking_result['camera'] if camera_id is None else self.fixed_cameras[camera_id - 1]

            azim = self.config['camera_azimuth']
            elev = self.config['camera_elevation']

            if len(avatar.blend_weights) > 1:
                blend_step = 0.04
                avatar.blend_weights[-2] = max(0.0, avatar.blend_weights[-2] - blend_step)
                avatar.blend_weights[-1] = min(1.0, avatar.blend_weights[-1] + blend_step)

            # camera.T[0, 1] += .2

            new_camera = orbit_camera(
                camera,
                azim=azim,
                elev=elev,
                roll=0.0,
                origin=(0, 0, -0.25)
            )

            render = avatar.render(new_camera)

            stats = dict(fps=avatar.fps())
            try:
                self.display_queue.put((full_frame, tracked_image, ref_img, render.copy(), stats, time.time()))
            except:
                print("Failed to enqueue output")

            # latency = time.time() - timestamp
            # print(f"Latency: {latency*1000:.1f} ms")

            if self.config['output'] == "cwipc":
                pc = convert_to_nparray(avatar.pcs[0])
                try:
                    self.pc_queue.put((pc, time.time()))
                except:
                    print("Failed to enqueue output")

            try:
                self.output_queue.put((render.copy(), time.time()))
            except:
                print("Failed to enqueue output")


class GUI(mp.Process):
    def __init__(self, display_queue, control_queue, config):
        super().__init__()
        self.display_queue = display_queue
        self.control_queue = control_queue
        self.config = config
        self.quit = False
        self.azim = 0.0
        self.elev = 0.0
        self.dist = 1.0
        self.dragging = False
        self.last_x, self.last_y = 0, 0

    def set_ref_img(self, ref_img_id: int | None):
        ref_img = None
        self.config['ref_img_id'] = ref_img_id
        if ref_img_id is not None:
            ref_img_path = self.config['ref_img_paths'][ref_img_id]
            print(f"Loading face from {ref_img_path}")
            img = cv2.imread(ref_img_path)
            if img is None:
                print(f"Image file not found: {ref_img_path}!")
                return
            ref_img = pad_to_square(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        try:
            self.control_queue.put(({'set_ref_img': ref_img}, time.time()))
        except:
            print("Failed to enqueue GUI command")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.last_x, self.last_y = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            dx = x - self.last_x
            dy = y - self.last_y
            self.last_x, self.last_y = x, y
            self.azim -= dx * 0.5
            self.elev -= dy * 0.5

        elif event == cv2.EVENT_MOUSEWHEEL:
            print("mousewheel", self.dist)
            if flags > 0:
                self.dist *= 0.9
            else:
                self.dist *= 1.1

    def handle_key_press(self, key: int):

        print(key)
        if key <= 0:
            return

        num_ref_imgs = len(self.config['ref_img_paths'])
        ref_img_id = self.config['ref_img_id']

        if key == ord('c'):
            self.set_ref_img(None)
        elif key == ord('0'):
            self.config['camera_id'] = None
        elif 1 <= key - ord('0') <= 9:
            self.config['camera_id'] = key - ord('0')
        elif key == 32:  # space
            if ref_img_id is None:
                ref_img_id = -1
            self.set_ref_img((ref_img_id + 1) % num_ref_imgs)
        elif key == 81:  # left
            if ref_img_id is None:
                ref_img_id = 0
            self.set_ref_img((ref_img_id - 1) % num_ref_imgs)
        elif key == 83:  # right
            if ref_img_id is None:
                ref_img_id = -1
            self.set_ref_img((ref_img_id + 1) % num_ref_imgs)
        elif key == 82:  # up
            pass
        elif key == 84:  # down
            pass
        elif key == ord('q'):
            self.quit = True
        elif key == ord('a'):
            self.config['camera_azimuth'] -= 4.0
        elif key == ord('d'):
            self.config['camera_azimuth'] += 4.0
        elif key == ord('w'):
            self.config['camera_elevation'] -= 4.0
        elif key == ord('s'):
            self.config['camera_elevation'] += 4.0
        elif key == 43:
            self.config['camera_distance'] *= 0.5
        elif key == 45:
            self.config['camera_distance'] *= 1.5

    def run(self):
        cv2.namedWindow("LiveAvatar")
        cv2.setMouseCallback("LiveAvatar", self.mouse_callback)

        while not self.quit:
            try:
                full_frame, tracked_image, ref_img, render, stats, timestamp = self.display_queue.get()
            except:
                continue

            canvas = np.ones((512, 4 * 256, 3), dtype=np.uint8) * int(0.1 * 255)

            new_width = 256
            new_height = int(new_width / (full_frame.shape[1] / full_frame.shape[0]))

            img_frame = cv2.resize(full_frame, dsize=(new_width, new_height), interpolation=cv2.INTER_CUBIC)
            if self.config['detect_face']:
                img_thumbnail = cv2.resize(tracked_image.copy(), dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
                paste_image(img_frame, img_thumbnail, 0, img_frame.shape[1] - img_thumbnail.shape[1])

            # paste_image(canvas, img_frame, 0, 0)

            img_avatar = cv2.resize(render.copy(), dsize=(460, 460), interpolation=cv2.INTER_CUBIC)
            # img_avatar = cv2.resize(render, dsize=(448, 448), interpolation=cv2.INTER_CUBIC)
            # img_input = cv2.resize(tracked_image.copy(), dsize=(256, 256), interpolation=cv2.INTER_CUBIC)

            img_avatar = cv2.putText(img_avatar, f"{stats['fps']:.2f}", (2, 18),
                                     cv2.FONT_HERSHEY_DUPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

            paste_image(
                canvas,
                img_avatar,
                # canvas.shape[0]-img_avatar.shape[0]-1,
                # canvas.shape[0]//2-img_avatar.shape[0]//2,
                10,
                canvas.shape[1] // 2 - img_avatar.shape[1] // 2
            )

            # paste_image(canvas, img_input, canvas.shape[0]-img_input.shape[0], 0)
            paste_image(canvas, img_frame, canvas.shape[0] - img_frame.shape[0], 0)

            if ref_img is not None:
                img_ref = cv2.resize(ref_img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
                paste_image(canvas, img_ref, canvas.shape[0] - img_ref.shape[0], canvas.shape[1] - img_ref.shape[1])

            show_image("LiveAvatar", canvas)
            key = cv2.waitKey(1)

            if key >= 0:
                self.handle_key_press(key)

            self.config['camera_azimuth'] = self.azim
            self.config['camera_elevation'] = self.elev
            # self.config['camera_distance'] = self.dist


class VideoSource(mp.Process):

    def __init__(
            self,
            frame_queue: LastOnlyQueue,
            video_path: str,
            video_size: tuple[int, int] | None = None,
            fps: float | None = None,
            loop: bool = False,
            show: bool = False
    ):
        super().__init__()
        self.frame_queue = frame_queue
        self.video_path = video_path
        self.show = show
        self.video_size = video_size
        self.fps = fps
        self.progress = False
        self.loop = loop
        self.time_last_frame = time.time()

    def run(self):

        # if not os.path.isfile(self.video_path):
        #     raise IOError(f"Invalid video file {self.video_path}")

        video, frame_rate = get_images(self.video_path)

        if self.fps is None:
            self.fps = frame_rate

        frame_duration = 1.0 / self.fps if self.fps > 0 else 0

        while True:
            print(f"Playing video {self.video_path}...")

            for fid in tqdm.tqdm(range(len(video)), disable=not self.progress):
                frame_rgb = video[fid]

                if self.video_size is not None:
                    frame_rgb = pad_to_square(frame_rgb)
                    frame_rgb = cv2.resize(frame_rgb, dsize=self.video_size)

                wait_sec = self.time_last_frame + frame_duration - time.time()
                if wait_sec > 0:
                    time.sleep(wait_sec)
                self.time_last_frame = time.time()

                if self.frame_queue is not None:
                    # try:
                    self.frame_queue.put((frame_rgb, time.time()))
                    # except:
                    #     print("Queue full. Dropping frame.")

                if self.show:
                    show_image("Video input", frame_rgb)
                    cv2.waitKey(1)

            if not self.loop:
                break


class VirtualCamera(mp.Process):

    def __init__(self, output_queue):
        super().__init__()
        self.output_queue = output_queue

    def run(self):
        import pyvirtualcam

        width = 512
        height = 512
        # width = 1280
        # height = 720
        with pyvirtualcam.Camera(width=width, height=height, fps=20) as cam:
            print(f'Using virtual camera: {cam.device}')
            while True:
                frame, timestamp = self.output_queue.get()
                if frame is None:
                    break
                frame = cv2.resize(frame, dsize=(width, height)).astype(np.uint8)
                # frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                # frame[:, :, 0] = np.linspace(0, 255, 720)[:, None]  # Red channel
                cam.send(frame)
                cam.sleep_until_next_frame()
                show_image("virtualcam", frame, wait=1)


def main(cfg):
    mp.set_start_method("spawn")

    frame_queue = LastOnlyQueue()
    input_queue = LastOnlyQueue()
    output_queue = LastOnlyQueue()
    pc_queue = LastOnlyQueue()
    display_queue = LastOnlyQueue()
    control_queue = mp.Queue()

    receiver = None
    sender = None
    gui = None

    with mp.Manager() as manager:
        config = manager.dict(cfg.__dict__)

        tracker = TrackWorker(frame_queue, input_queue, show=cfg.gui, detect_face=cfg.detect_face, config=config)
        tracker.start()

        if cfg.gui:
            gui = GUI(display_queue, control_queue, config)
            gui.start()

        worker = Worker(input_queue, output_queue, display_queue, pc_queue, control_queue,
                        checkpoint=cfg.checkpoint, device=cfg.device, show=cfg.gui, config=config)
        worker.start()

        if cfg.input == "video":
            receiver = VideoSource(frame_queue, cfg.video_file, loop=cfg.loop, video_size=cfg.resize)
            receiver.start()
        elif cfg.input == "webrtc":
            pass # FIXME: add param to enable/disable send/receive
            # webrtc.run_webrtc_server(frame_queue, output_queue)
        elif cfg.input == "socket":
            receiver = ImageSocketReceiver(frame_queue, cfg.host, cfg.port_recv)
            receiver.start()
        elif cfg.input == "webcam":
            receiver = WebcamGrabber(frame_queue)
            receiver.start()
        else:
            raise ValueError("Unknown input mode.")

        if cfg.output == "webrtc":
            pass
            # webrtc.run_webrtc_server(frame_queue, output_queue)
        elif cfg.output == "socket":
            sender = ImageSocketSender(output_queue, cfg.host, cfg.port_send)
            sender.start()
        elif cfg.output == "virtualcam":
            sender = VirtualCamera(output_queue)
            sender.start()
        elif cfg.output == "cwipc":
            from server.cwipc_utils import CWIPCSocketSender, convert_to_cwipc, convert_to_nparray
            sender = CWIPCSocketSender(pc_queue, cfg.host, port=4303)
            sender.start()
        else:
            pass

        if tracker is not None:
            tracker.join()

        if worker is not None:
            worker.join()

        if receiver is not None:
            receiver.join()

        if sender is not None:
            sender.join()

        if gui is not None:
            gui.join()


if __name__ == "__main__":

    import sys
    import signal

    def handler(signum, frame):
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)

    @dataclass
    class Config():
        checkpoint: str | None = "vitb_vits_p24_l4_pxsh_180_sh0/epoch_00724"
        image_size: int = 336
        render_size: int = 512
        pred_expr: bool = True

        input: Literal['video', 'webcam', 'socket', 'webrtc'] = "webcam"
        output: Literal['none', 'socket', 'webrtc', 'virtualcam', 'cwipc'] = "none"
        host: str = "127.0.0.1"
        port_recv: int = 9000
        port_send: int = 9001

        gui: bool = True
        device: str = 'cuda'

        video_file: str | None = None

        loop: bool = True
        resize: tuple[int, int] | None = (512, 512)

        detect_face: bool = input == "webcam"

        camera_id: int | None = None
        camera_azimuth: float = 0.0
        camera_elevation: float = 0.0
        camera_distance: float = 1.0
        alpha: float = 0.0
        ref_img_id: int | None = None
        ref_img_paths: tuple[str, ...] = (
            "assets/demo_faces/mona-lisa.png",
            "assets/demo_faces/vincent.png",
        )


    cfg = tyro.cli(Config)

    main(cfg)
