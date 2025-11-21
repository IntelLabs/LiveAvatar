import socket
import time
import struct
import cv2
import numpy as np
import multiprocessing as mp


def encode_jpeg(image: np.ndarray, jpeg_quality: int | None = None) -> bytes:
    params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality] if jpeg_quality is not None else None
    return cv2.imencode('.jpg', image, params)[1].tobytes()


def decode_jpeg(jpeg_data: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(jpeg_data, np.uint8), cv2.IMREAD_COLOR)


def socket_recv_size(socket, size):
    data = b''
    while len(data) < size:
        packet = socket.recv(size - len(data))
        if not packet:
            return None
        data += packet
    return data


class ImageSocketReceiver(mp.Process):

    def __init__(self, frame_queue, host: str, port: int, show=False):
        super().__init__()
        self.frame_queue = frame_queue
        self.host = host
        self.port = port
        self.show = port

    def _recv(self, sock, size):
        data = b''
        while len(data) < size:
            packet = sock.recv(size - len(data))
            if not packet:
                return None
            data += packet
        return data

    def run(self):

        print(f"Receiver listening on {self.host}:{self.port}...")

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((self.host, self.port))
        server_socket.listen(1)

        try:
            while True:
                conn, addr = server_socket.accept()
                print(f"Receiver connected to {addr}")

                try:
                    while True:
                        length_data = self._recv(conn, 4)
                        if not length_data:
                            break
                        length = int.from_bytes(length_data, byteorder='big')

                        image_data = self._recv(conn, length)
                        if not image_data:
                            break

                        frame = decode_jpeg(image_data)

                        self.frame_queue.put((frame, time.time()))

                        if self.show:
                            cv2.imshow("[Server] Received image", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                            cv2.waitKey(1)

                except (ConnectionResetError, BrokenPipeError):
                    print(f"Disconnected from {addr}")
                    conn.close()

        except Exception as e:
            print(f"Socket receiver error: {e}")

        finally:
            server_socket.close()
            cv2.destroyAllWindows()
            print("Socket receiver shut down")



class ImageSocketSender(mp.Process):

    def __init__(self, output_queue, host: str, port: int):
        super().__init__()
        self.output_queue = output_queue
        self.host = host
        self.port = port

    def run(self):

        print(f"Sender waiting for client connection on {self.host}:{self.port}...")

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((self.host, self.port))
        server_socket.listen(1)

        try:
            while True:
                conn, addr = server_socket.accept()
                print(f"Sender connected to {addr}")

                try:
                    while True:
                        frame, timestamp = self.output_queue.get()
                        if frame is None:
                            break

                        jpeg_bytes = encode_jpeg(frame)
                        msg_len = struct.pack('>I', len(jpeg_bytes))
                        conn.sendall(msg_len + jpeg_bytes)

                except (ConnectionResetError, BrokenPipeError):
                    print(f"Disconnected from {addr}")
                    conn.close()

        except Exception as e:
            print(f"Socket sender error: {e}")
        finally:
            server_socket.close()
            print("Socket sender shut down")
