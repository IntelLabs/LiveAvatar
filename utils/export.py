import os
import numpy as np
from plyfile import PlyData, PlyElement
import torch

from models.gaussian_pointclouds import GaussianPointclouds, PointcloudConverter, create_triangle_faces, map_to_points
from utils.util import create_camera, to_numpy


def _write_ply(file_path,
               xyz: np.ndarray,
               rgb: np.ndarray,
               opacities: np.ndarray,
               scales: np.ndarray,
               rots: np.ndarray,
               faces: np.ndarray | None = None,
               binary: bool = False):
    """
    Write a PLY file for a mesh or a point cloud.

    :param coords: an [N x 3] array of floating point coordinates.
    :param rgb: an [N x 3] array of vertex colors, in the range [0.0, 1.0].
    :param faces: an [N x 3] array of triangles encoded as integer indices.
    """

    extension = '.ply'

    out_dir = os.path.split(file_path)[0]
    if out_dir != '':
        os.makedirs(out_dir, exist_ok=True)

    if file_path[-4:] != extension:
        file_path = file_path + extension

    vertex_data = np.hstack([
        xyz,
        rgb,
        opacities,
        scales,
        rots,
    ])
    vertex_data = [tuple(v) for v in vertex_data]
    vertex_dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        # ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')
    ]
    vertex = np.array(vertex_data, dtype=vertex_dtype)
    elv = PlyElement.describe(vertex, 'vertex')
    elements = [elv]

    if faces is not None:
        faces = np.array(
            [(f,) for f in faces],
            dtype=[('vertex_indices', 'i4', (3,))]
        )
        elf = PlyElement.describe(faces, 'face')
        elements.append(elf)

    PlyData(elements, text=not binary).write(file_path)


def write_ply(file_path: str, pc: GaussianPointclouds, with_faces=False):
    camera = create_camera(0, 0, fov=30).to(pc.device)

    xyz = pc.get_xyz()  # (N, 3)
    rgb = pc.get_colors(camera.get_camera_center())  # (N, 3)

    faces = None
    if with_faces:
        faces = create_triangle_faces(pc._theta_dim, pc._phi_dim)
        faces = faces.repeat([len(pc), 1])

    _write_ply(file_path,
               to_numpy(xyz),
               to_numpy(rgb),
               to_numpy(map_to_points(pc.opacity)),
               to_numpy(map_to_points(pc.scaling)),
               to_numpy(map_to_points(pc.rotation)),
               faces=to_numpy(faces),
               binary=True)


def read_ply(file_path: str) -> GaussianPointclouds:
    plydata = PlyData.read(file_path)
    h = int(len(plydata.elements[0].data['x'])**0.5)+1

    x = plydata.elements[0].data['x']
    y = -plydata.elements[0].data['y']
    z = plydata.elements[0].data['z']
    r = plydata.elements[0].data['f_dc_0']
    g = plydata.elements[0].data['f_dc_1']
    b = plydata.elements[0].data['f_dc_2']
    o = plydata.elements[0].data['opacity']
    sx = plydata.elements[0].data['scale_0']
    sy = plydata.elements[0].data['scale_0']
    sz = plydata.elements[0].data['scale_0'] # * 0 + np.log(0.0001)
    rot0 = plydata.elements[0].data['rot_0']
    rot1 = plydata.elements[0].data['rot_1']
    rot2 = plydata.elements[0].data['rot_2']
    rot3 = plydata.elements[0].data['rot_3']


    print(sx[1], sy[1], sx[1])

    print(x.mean(), y.mean(), z.mean())

    x = x - x.mean()
    y = y - y.mean()
    z = z - z.mean()
    # z -= 3.5

    def inverse_sigmoid(x):
        return np.log(x / (1 - x))

    # o[:] = inverse_sigmoid(0.9)

    data = np.stack([
        x, y, z,
        o,
        sx, sy, sz,
        rot0, rot1, rot2, rot3,
        r, g, b
    ])
    # data[:3] /= 2.0
    # data[4:7] -= 1.0

    data[:3] /= 15.
    data[4:7] -= 5.0

    data[7] = 1.0
    data[8:11] = 0.0

    features = np.zeros((1, 14, h*h), dtype=np.float32)
    features[0, 4:7] = np.log(0.00001)
    features[0, :, :data.shape[1]] = data
    # features[0, 4:7] = np.log(0.001)

    features = features.reshape(1, 14, h, h)

    pc = GaussianPointclouds(torch.tensor(features, device="cuda"))
    return pc


if __name__ == '__main__':
    # plydata = PlyData.read('../assets/tet.ply')
    # print(plydata)

    xyz = np.array([(0, 0, 0),
                    (0, 1, 1,),
                    (1, 0, 1,),
                    (1, 1, 0)])
    rgb = np.array([(0, 0, 0),
                    (0, 1, 1,),
                    (1, 0, 1,),
                    (1, 1, 0)])
    scales = np.ones(len(xyz)) * 10

    faces = np.array([([0, 1, 2]),
                      ([0, 2, 3]),
                      ([0, 1, 3]),
                      ([1, 2, 3])])

    # face_data = [(f,) for f in face_data]
    # face = np.array(face_data, dtype=[('vertex_indices', 'i4', (3,))])

    # elf = PlyElement.describe(face, 'face')
    write_ply("out.ply", xyz, rgb, faces=faces)

