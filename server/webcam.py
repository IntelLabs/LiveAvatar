import cv2
import time
import multiprocessing as mp


class WebcamGrabber(mp.Process):

    def __init__(
            self,
            frame_queue,
            video_size: tuple[int, int] | None = None,
            fps: float | None = None,
            show: bool = False
    ):
        super().__init__()
        self.frame_queue = frame_queue
        self.show = show
        self.video_size = video_size
        self.fps = fps

    def run(self):
        print("Starting webcam...")
        cap = cv2.VideoCapture(0)
        if self.video_size is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_size[1])
        if self.fps is not None:
            cap.set(cv2.CAP_PROP_FPS, self.fps)

        if not cap.isOpened():
            print("Cannot open webcam")
            return

        try:
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    print("Failed to read frame")
                    break

                frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                if self.frame_queue is not None:
                    try:
                        self.frame_queue.put((frame, time.time()))
                    except:
                        print("Queue full. Dropping frame.")

                if self.show:
                    cv2.imshow("Webcam input", frame_bgr)
                    cv2.waitKey(1)
        finally:
            print("Webcam stopped")
            cap.release()
            if self.show:
                cv2.destroyAllWindows()


if __name__ == '__main__':
    import sys
    import signal

    def handler(signum, frame):
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)

    webcam_grabber = WebcamGrabber(None, video_size=(1280, 720), show=True)
    webcam_grabber.start()
    webcam_grabber.join()