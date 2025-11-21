from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from pytorch3d.renderer import TexturesVertex, FoVPerspectiveCameras
from pytorch3d.structures import Meshes


from utils.sh_utils import eval_sh
from utils.sh_utils import RGB2SH
from utils.util import PolarToCartesian, log_mapping, create_camera

# SCALE_INIT = 0.0075 * 0.5
# SCALE_INIT = 0.0025
SCALE_INIT = 0.0020


def points_to_map(data: torch.Tensor,
                  nchannels: int | None = None,
                  height: int | None = None,
                  width: int | None = None):
    if len(data.shape) == 3 and data.shape[0] > 1000:
        data = data.reshape(data.shape[0], -1)
    in_shape = data.shape
    if len(data.shape) == 2:
        data = data.unsqueeze(0)
    B, K, C = data.shape
    if nchannels is None:
        nchannels = C
    if height is None and width is None:
        height, width = [int(K**0.5)] * 2
    assert height * width == K, "invalid number of points or incorrect height/width"
    map = data[:, :, :nchannels].reshape((B, height, width, nchannels)).permute(0, 3, 1, 2)
    if len(in_shape) == 2:
        map = map[0]
    return map


def map_to_points(map: torch.Tensor):
    B, C, H, W = map.shape
    return map.permute(0, 2, 3, 1).reshape(-1, C)


def inverse_sigmoid(x):
    return torch.log(x/(1-x))


def create_sphere(w, h, radius=1.0, k_phi=None, k_theta=None, scales=None, azimuth_range=135):
    # Create grid
    theta = torch.linspace(0.00, np.pi * 0.9, h)
    theta = log_mapping(theta, k_theta)

    # phi = torch.linspace(0, 2*np.pi, w).cuda()
    # phi = torch.linspace(np.pi * 0.25, 2*np.pi-np.pi*0.25, w).cuda()
    # phi = torch.linspace(np.pi * 0.5, 2*np.pi-np.pi*0.5, w).cuda()
    phi = torch.linspace(np.radians(180-azimuth_range), np.radians(180+azimuth_range), w)
    phi = log_mapping(phi, k_phi)

    phi_grid, theta_grid = torch.meshgrid(phi, theta, indexing='xy')

    if scales is None:
        scales = (0, 0, 0)

    # Spherical to Cartesian coordinates
    x = radius * torch.sin(theta_grid) * torch.cos(phi_grid) * (scales[0] + 1.0)
    y = radius * torch.sin(theta_grid) * torch.sin(phi_grid) * (scales[1] + 1.0)
    z = radius * torch.cos(theta_grid) * (scales[2] + 1.0)

    # Apply rotation to align with z-axis
    # Rotation matrix for 90 degrees around y-axis
    rotation_matrix_y = torch.Tensor([
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0]
    ])
    rotation_matrix_z = torch.Tensor([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 1]
    ])
    rotated_coords = rotation_matrix_z @ -rotation_matrix_y @torch.stack([x.ravel(), y.ravel(), z.ravel()])
    x, y, z = rotated_coords.reshape(3, *x.shape)
    # z = z - 0.30
    return torch.dstack([x, y, z]).reshape(-1, 3)


def gaussian_unit_sphere(w, h, radius=0.25, color=0.5, sh_degree=0, k_phi=None, k_theta=None, azimuth_range=135,
                         opacity=0.5):
    points = create_sphere(w, h, radius=radius, k_phi=k_phi, k_theta=k_theta, azimuth_range=azimuth_range)
    colors = torch.ones_like(points) * color

    features = create_feature_map_from_points(
        points=points, colors_rgb=colors, W=w, H=h, sh_degree=sh_degree, opacity=opacity
    )
    return features


def create_feature_map_from_points(points, colors_rgb, W, H, sh_degree, opacity):
    xyz = points_to_map(points)

    use_knn_estimate = False

    if use_knn_estimate:
        from simple_knn._C import distCUDA2
        dist2 = torch.clamp_min(distCUDA2(points.cuda()).cpu(), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        scales = points_to_map(scales)
    else:
        # avg_dist = 0.0002**0.5
        # dist = torch.ones((3, H, W)) * avg_dist * 7
        dist = torch.ones((3, H, W)) * SCALE_INIT
        scales = torch.log(dist)

    shs = torch.zeros((3 * (sh_degree + 1)**2, H, W))
    shs[:3] = points_to_map(RGB2SH(colors_rgb))

    rots = torch.zeros(4, H, W)
    rots[:, 0] = 1

    opacities = inverse_sigmoid(opacity * torch.ones((1, H, W)))

    return torch.cat([
        xyz.cpu(),
        opacities,
        scales,
        rots,
        shs
    ])


def build_color(means3D, shs, camera_position, sh_degree):
    rays_o = camera_position
    rays_d = means3D - rays_o
    rays_d = rays_d / torch.norm(rays_d)
    color = eval_sh(sh_degree, shs, rays_d)
    return (color + 0.5).clamp(min=0.0, max=1.0)


class GPCParams():

    def __init__(self, sh_degree, polar, k_phi, k_theta):
        self.sh_degree = sh_degree
        self.polar = polar
        self.k_phi = k_phi
        self.k_theta = k_theta
        self.num_sh_channels = (sh_degree+1)**2 * 3
        self.layers = [
            # name, num channels, channel rescaling factor
            ('coords', 1 if polar else 3, 0.30),
            ('opacity', 1, 4.0),
            # ('scaling', 3, 2.5),
            ('scaling', 3, 3.5),
            ('rotation', 4, 1.0),
            ('shs', self.num_sh_channels, 2.0),
        ]
        self.channels = {}
        self.scale_factors = {}
        st = 0
        for name, nc, scale in self.layers:
            self.channels[name] = slice(st, st+nc)
            self.scale_factors[name] = scale
            st = st + nc

    @property
    def num_coord_dims(self): return self.layers[0][1]

    @property
    def num_opacity_dims(self): return self.layers[1][1]

    @property
    def num_scale_dims(self): return self.layers[2][1]

    @property
    def num_rotation_dims(self): return self.layers[3][1]

    @property
    def num_shs_dims(self): return self.layers[4][1]

    def num_dims(self): return sum([l[1] for l in self.layers])

    @property
    def coord_channels(self): return self.channels['coords']

    @property
    def opacity_channels(self): return self.channels['opacity']

    @property
    def scale_channels(self): return self.channels['scaling']

    @property
    def rotation_channels(self): return self.channels['rotation']

    @property
    def shs_channels(self): return self.channels['shs']


class GaussianPointclouds(nn.Module):
    _polar_to_cartesian = None

    def __init__(
            self,
            features: torch.Tensor,
            pos: torch.Tensor = None,
            scales: torch.Tensor = None,
            params: GPCParams = None,
    ):
        super().__init__()

        assert len(features.shape) == 4
        self._features = features

        if pos is None:
            pos = torch.zeros((len(features), 1), device=features.device)

        if scales is None:
            scales = torch.ones((1, 3), device=features.device)

        self._pos = pos
        self._scales = scales

        self._theta_dim = features.shape[2]
        self._phi_dim = features.shape[3]

        self.scaling_activation = torch.exp
        self.opacity_activation = torch.sigmoid
        self.rotation_activation = torch.nn.functional.normalize

        if params is None:
            params = GPCParams(0, False, k_theta=0.5, k_phi=1.0)  # FIXME: make sure to get correct parameters

        self._params = params

    @property
    def device(self):
        return self._features.device

    def cartesian(self, x):
        if not self._params.polar:
            return x

        if GaussianPointclouds._polar_to_cartesian is None:
            GaussianPointclouds._polar_to_cartesian = PolarToCartesian(
                self._theta_dim, self._phi_dim,
                k_phi=self._params.k_phi, k_theta=self._params.k_theta,
                device=self.device
            )

        return GaussianPointclouds._polar_to_cartesian(x, self._scales)


    def __len__(self):
        return self._features.shape[0]

    @property
    def sh_degree(self):
        return self._params.sh_degree

    @property
    def xyz(self):
        xyz = self.cartesian(self._features[:, self._params.coord_channels])
        scales = self._scales.reshape(-1, 3, 1, 1)
        xyz = xyz * scales
        # xyz[:, 0] *= scales[:, 0]
        # xyz[:, 1] *= scales[:, 1]
        # xyz[:, 2] *= scales[:, 2]
        xyz = xyz + self._pos.unsqueeze(-1).unsqueeze(-1)
        return xyz

    @property
    def opacity(self):
        return self._features[:, self._params.opacity_channels]

    @property
    def scaling(self):
        return self._features[:, self._params.scale_channels]

    @property
    def rotation(self):
        return self._features[:, self._params.rotation_channels]

    @property
    def shs(self):
        return self._features[:, self._params.shs_channels]

    def get_xyz(self, return_map: bool = False) -> torch.Tensor:
        if return_map:
            return self.xyz
        return map_to_points(self.xyz)

    def get_opacity(self, return_map: bool = False) -> torch.Tensor:
        opacity = self.opacity_activation(self.opacity)
        if return_map:
            return opacity
        return map_to_points(opacity)

    def get_scaling(self, return_map: bool = False) -> torch.Tensor:
        scaling = self.scaling_activation(self.scaling)
        if return_map:
            return scaling
        return map_to_points(scaling)

    def get_rotation(self, return_map: bool = False) -> torch.Tensor:
        rot = self.rotation_activation(self.rotation)
        if return_map:
            return rot
        return map_to_points(rot)

    def get_colors(self, camera_position, return_map: bool = False) -> torch.Tensor:
        points = map_to_points(self.xyz)
        shs = map_to_points(self.shs)
        shs = shs.reshape(shs.shape[0], -1, 3).permute(0, 2, 1)
        colors_rgb = build_color(
            points, shs=shs, camera_position=camera_position, sh_degree=self.sh_degree
        )
        if return_map:
            colors_rgb = points_to_map(colors_rgb)
        return colors_rgb

    # def to_mesh(self, fov=30):
    #     convert_to_meshes = PointcloudConverter(
    #         batchsize=self.cfg.batchsize,
    #         w=self..params.feature_map_size,
    #         h=self.net.params.feature_map_size,
    #         device=self.net.device
    #     )
    #     create_camera(45, 0, fov=fov).to(self.device)
    #     mesh = convert_to_meshes(self)

    def __getitem__(self, key: int | slice):
        if isinstance(key, slice):
            start, stop, step = key.start, key.stop, key.step
        elif isinstance(key, int):
            start, stop, step = key, key+1, 1
        else:
            raise TypeError(
                f"{type(self).__name__} indices must be integers or slices, not {type(key).__name__}"
            )
        return GaussianPointclouds(
            features=self._features[start:stop:step],
            pos=self._pos[start:stop:step],
            scales=self._scales[start:stop:step],
            params=self._params,
        )


def create_triangle_faces(h: int, w: int) -> torch.Tensor:
    # Generate grid indices
    indices = torch.arange(h * w, dtype=torch.int32).reshape(h, w)

    # Create faces (triangles)
    faces = []
    for i in range(h - 1):
        for j in range(w - 1):
            # Define the vertices for the current cell
            v0 = indices[i, j]
            v1 = indices[i, j + 1]
            v2 = indices[i + 1, j]
            v3 = indices[i + 1, j + 1]

            # Add two triangles
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
            # faces.append([v2, v1, v3])

            # faces.append([v0, v2, v1])
            # faces.append([v1, v2, v3])

    connect_horizontal_edges = False
    if connect_horizontal_edges:
        j = w - 1
        for i in range(h - 1):
            v0 = indices[i, j]
            v1 = indices[i, 0]
            v2 = indices[i + 1, j]
            v3 = indices[i + 1, 0]
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])

    return torch.tensor(faces, dtype=torch.int32)


def to_pytorch3d_mesh(points: torch.Tensor, colors: torch.Tensor) -> Meshes:
    grid_size = int(np.sqrt(points.shape[0]))
    faces = create_triangle_faces(grid_size, grid_size).to(points.device)
    textures = TexturesVertex(verts_features=colors.unsqueeze(0))
    return Meshes(verts=[points], faces=[faces], textures=textures)


class PointcloudConverter():
    def __init__(self, batchsize, w, h, device):
        self._faces = create_triangle_faces(w, h).to(device)
        self._faces = self._faces.unsqueeze(0).repeat([batchsize, 1, 1])

    def _get_points_from_pointclouds(self, pointclouds: GaussianPointclouds) -> torch.Tensor:
        N = len(pointclouds)
        return pointclouds.xyz.reshape(N, 3, -1).permute(0, 2, 1)  # [N, 3, H, W] -> [N, V, 3]

    def _get_colors_from_pointclouds(
            self,
            pointclouds: list[GaussianPointclouds],
            cameras: FoVPerspectiveCameras
    ) -> torch.Tensor:
        colors = []
        for i in range(len(pointclouds)):
            cl = pointclouds[i].get_colors(cameras[i].get_camera_center()[0])
            colors.append(cl)
        return torch.stack(colors)

    def __call__(self, pointclouds, cameras, points=None, with_colors=False) -> Meshes:
        if points is None:
            points = self._get_points_from_pointclouds(pointclouds)
        if with_colors:
            colors = self._get_colors_from_pointclouds(pointclouds, cameras)
        else:
            colors = torch.ones_like(points)
        n_meshes = points.shape[0]
        textures = TexturesVertex(verts_features=colors)
        return Meshes(verts=points, faces=self._faces[:n_meshes], textures=textures)

