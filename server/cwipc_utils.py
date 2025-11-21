import socket
import time
import struct
import cv2
import numpy as np
import multiprocessing as mp

import cwipc

from models.gaussian_pointclouds import GaussianPointclouds
from utils.util import create_camera, to_numpy


def convert_to_nparray(pc: GaussianPointclouds):

    camera = create_camera(0, 0, fov=30).to(pc.device)

    # xyz = to_numpy(pc.get_xyz())  # (N, 3)
    # rgb = to_numpy(pc.get_colors(camera.get_camera_center()))  # (N, 3)
    #
    # vertex_data = np.hstack([xyz, rgb * 255.0, np.zeros_like(xyz[:, 0:1])])
    # vertex_data = [tuple(v) for v in vertex_data]
    # vertex_dtype = [
    #     ('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('r', 'u1'), ('g', 'u1'), ('b', 'u1'), ('tile', 'u1')
    # ]
    # np_array = np.array(vertex_data, dtype=vertex_dtype)

    xyz = to_numpy(pc.get_xyz())  # (N, 3)
    rgb = to_numpy(pc.get_colors(camera.get_camera_center()))  # (N, 3)

    # Define the structured array dtype
    vertex_dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('r', 'u1'), ('g', 'u1'), ('b', 'u1'),
        ('tile', 'u1')
    ]

    # Create the structured array directly
    N = xyz.shape[0]
    np_array = np.zeros(N, dtype=vertex_dtype)

    # Assign data directly to the fields (no loops, no tuples)
    np_array['x'] = xyz[:, 0]
    np_array['y'] = xyz[:, 1]
    np_array['z'] = xyz[:, 2]
    np_array['r'] = (rgb[:, 0] * 255.0).astype('u1')
    np_array['g'] = (rgb[:, 1] * 255.0).astype('u1')
    np_array['b'] = (rgb[:, 2] * 255.0).astype('u1')
    np_array['tile'] = 0  # All zeros, already initialized

    return np_array


def convert_to_cwipc(pc: GaussianPointclouds):
    np_array = convert_to_nparray(pc)
    return cwipc.cwipc_from_numpy_array(np_array, 0)


import struct
from typing import Union, Callable

CHUNK = 1024  # bytes
FOURCC = "cwi0"


# 4CC handling doesn't really belong here, but it's convenient.
vrt_fourcc_type = Union[int, bytes, str]

def VRT_4CC(code : vrt_fourcc_type) -> int:
    """Convert anything reasonable (bytes, string, int) to 4cc integer"""
    if isinstance(code, int):
        return code
    if not isinstance(code, bytes):
        assert isinstance(code, str)
        code = code.encode('ascii')
    assert len(code) == 4
    rv = (code[0]<<24) | (code[1]<<16) | (code[2]<<8) | (code[3])
    return rv


def gen_header(data : bytes) -> bytes:
    datalen = len(data)
    timestamp = int(time.time() * 1000)
    return struct.pack("=LLQ", VRT_4CC(FOURCC), datalen, timestamp)


def make_payload(pc=None) -> bytes:
    """Return the next chunk to send (bytes)."""
    if not pc:
        points = cwipc.cwipc_point_array(values=[(1, 2, 3, 0x10, 0x20, 0x30, 1), (4, 5, 6, 0x40, 0x50, 0x60, 2)])
        pc = cwipc.cwipc_from_points(points, 0)
    cpc = pc.get_packet()
    return cpc


class CWIPCSocketSender(mp.Process):

    def __init__(self, output_queue, host: str, port: int = 4303):
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

        path = "../assets/avatar.ply"
        pc_dummy = cwipc.cwipc_read(path, 0)

        try:
            while True:
                conn, addr = server_socket.accept()
                print(f"Sender connected to {addr}")

                try:
                    while True:
                        print("sending pointcloud...")
                        pc, timestamp = self.output_queue.get()
                        if pc is None:
                            break
                        pc = cwipc.cwipc_from_numpy_array(pc, 0)
                        # pc = convert_to_cwipc(gpc)

                        # jpeg_bytes = encode_jpeg(frame)
                        # msg_len = struct.pack('>I', len(jpeg_bytes))
                        # conn.sendall(msg_len + jpeg_bytes)

                        data = make_payload(pc)
                        hdr = gen_header(data)
                        packet = hdr + data
                        # print(f"streaming: {packet}")
                        conn.sendall(packet)
                        time.sleep(0.02)

                except (ConnectionResetError, BrokenPipeError):
                    print(f"Disconnected from {addr}")
                    conn.close()

        except Exception as e:
            print(f"Socket sender error: {e}")
        finally:
            server_socket.close()
            print("Socket sender shut down")
