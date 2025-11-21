import socket
import cv2
import time
import multiprocessing as mp

from server.util import LastOnlyQueue
from server.webcam import WebcamGrabber
from server.socket_io import encode_jpeg, decode_jpeg, socket_recv_size


class ImageClientSender(mp.Process):

    def __init__(self, queue, host: str, port: int, fps: float | None = None, show: bool = True):
        super().__init__()
        self.queue = queue
        self.host = host
        self.port = port
        self.show = show
        self.fps = fps

    def run(self):
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((self.host, self.port))
        print(f"Sender connected to server on {self.host}:{self.port}.")

        try:
            while True:
                frame, timestamp = self.queue.get()
                data = encode_jpeg(frame)
                length = len(data).to_bytes(4, byteorder='big')
                client_socket.sendall(length + data)

                if self.fps is not None:
                    time.sleep(1 / self.fps)
        except KeyboardInterrupt:
            print("Stopping sender...")
        finally:
            client_socket.close()


class ImageClientReceiver(mp.Process):

    def __init__(self, queue, host: str, port: int, show: bool = True):
        super().__init__()
        self.queue = queue
        self.host = host
        self.port = port
        self.show = show


    def run(self):
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((self.host, self.port))
        print(f"Receiver connected to server on {self.host}:{self.port}.")

        try:
            while True:
                length_data = socket_recv_size(client_socket, 4)
                if not length_data:
                    break
                length = int.from_bytes(length_data, byteorder='big')

                image_data = socket_recv_size(client_socket, length)
                if not image_data:
                    break

                frame = decode_jpeg(image_data)

                self.queue.put((frame, time.time()))

                if self.show:
                    cv2.imshow("[Client] Received image", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)

        except KeyboardInterrupt:
            print("Stopping receiver...")
        finally:
            client_socket.close()
            if self.show:
                cv2.destroyAllWindows()


def main(host: str = "127.0.0.1", port_send: int = 9000, port_recv: int = 9001):
    frame_queue = LastOnlyQueue()
    result_queue = LastOnlyQueue()

    webcam_grabber = WebcamGrabber(frame_queue)
    webcam_grabber.start()

    image_sender = ImageClientSender(frame_queue, host, port_send, show=True)
    image_sender.start()

    image_receiver = ImageClientReceiver(result_queue, host, port_recv, show=True)
    image_receiver.start()

    webcam_grabber.join()
    image_sender.join()
    image_receiver.join()


if __name__ == "__main__":
    main()