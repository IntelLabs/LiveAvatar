import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pytorch3d.structures import Meshes
from pytorch3d.loss import chamfer_distance
import itertools
import kornia

from configs.config import RegularizationWeights
from models.gaussian_pointclouds import GaussianPointclouds, GPCParams
# from utils.nn import to_numpy
# from visualization.vis import add_landmarks_to_images, draw_landmark_polygon, draw_landmarks
from configs.config import KP_IDS_FOREHEAD


def angular_difference(angle1, angle2, period=360):
    """
    Compute the smallest angular difference, considering periodicity.
    :param angle1: First angle in degrees or radians.
    :param angle2: Second angle in degrees or radians.
    :param period: Period of the angle (360 for degrees, 2*np.pi for radians).
    :return: Smallest angular difference.
    """
    diff = angle1 - angle2
    return (diff + period / 2) % period - period / 2


def laplacian_loss_polar(r, theta, phi, w_theta=1.0, w_phi=1.0, angle_period=360):
    """
    Compute the Laplacian smoothness loss for a trajectory in polar coordinates.
    :param r: Array of distances (N,).
    :param theta: Array of azimuthal angles (N,).
    :param phi: Array of elevation angles (N,).
    :param w_theta: Weight for azimuthal angle smoothness.
    :param w_phi: Weight for elevation angle smoothness.
    :param angle_period: Periodicity of the angles (360 for degrees, 2*pi for radians).
    :return: Scalar Laplacian loss.
    """
    # Differences for r
    laplacian_r = r[1:-1] - (r[:-2] + r[2:]) / 2
    loss_r = torch.sum(laplacian_r ** 2)

    # Differences for theta
    laplacian_theta = angular_difference(theta[1:-1], (theta[:-2] + theta[2:]) / 2, period=angle_period)
    loss_theta = w_theta * torch.sum(laplacian_theta ** 2)

    # Differences for phi
    laplacian_phi = angular_difference(phi[1:-1], (phi[:-2] + phi[2:]) / 2, period=angle_period)
    loss_phi = w_phi * torch.sum(laplacian_phi ** 2)

    # Total loss
    return loss_r + loss_theta + loss_phi


def laplacian_loss_conv(grid, kernel_size=3, circular=True):
    """
    Compute Laplacian loss using a larger kernel, supporting batches.

    Args:
        grid: Tensor of shape (B, 3, H, W) where each (B, :, i, j) stores a 3D coordinate (x, y, z).
        kernel_size: Size of the Laplacian kernel (3, 5, or 7).

    Returns:
        Scalar Laplacian loss.
    """
    assert kernel_size in [3, 5, 7], "Only supports 3x3, 5x5, or 7x7 kernels"

    # Define Laplacian kernels for different sizes
    laplacian_kernels = {
        3: torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32),
        5: torch.tensor([
            [0, 0, 1, 0, 0],
            [0, 1, 2, 1, 0],
            [1, 2, -12, 2, 1],
            [0, 1, 2, 1, 0],
            [0, 0, 1, 0, 0]
        ], dtype=torch.float32),
        7: torch.tensor([
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 2, 1, 0, 0],
            [0, 1, 2, 3, 2, 1, 0],
            [1, 2, 3, -20, 3, 2, 1],
            [0, 1, 2, 3, 2, 1, 0],
            [0, 0, 1, 2, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ], dtype=torch.float32),
    }

    c = grid.shape[1]

    kernel = laplacian_kernels[kernel_size].unsqueeze(0).unsqueeze(0)  # Shape (1, 1, k, k)
    kernel = kernel.repeat(c, 1, 1, 1).to(grid.device)

    # Apply circular padding before convolution
    if circular:
        pad = kernel_size // 2  # Amount of padding needed on each side
        grid_padded = F.pad(grid, (pad, pad, 0, 0), mode="circular")
    else:
        grid_padded = grid

    # Apply Laplacian smoothing independently to each channel (x, y, z)
    laplacian = F.conv2d(grid_padded, kernel, groups=c)

    # Apply Laplacian smoothing independently to each channel (x, y, z)
    # laplacian = F.conv2d(grid, kernel, padding=kernel_size // 2, groups=3)  # Depthwise convolution

    loss = torch.mean(laplacian ** 2)
    return loss


def mesh_surface_area(meshes: Meshes):
    verts = meshes.verts_packed()  # (V, 3)
    faces = meshes.faces_packed()  # (F, 3)

    # Get the triangle vertices
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    # Compute edge vectors
    edge1 = v1 - v0
    edge2 = v2 - v0

    # Compute cross product and area
    cross_product = torch.cross(edge1, edge2, dim=1)
    triangle_areas = 0.5 * torch.norm(cross_product, dim=1)

    # Sum all triangle areas to get total surface area
    surface_area = triangle_areas.sum()

    return surface_area


def chamfer_face_distance(pointclouds: GaussianPointclouds, landmarks3d: torch.Tensor, occluded: torch.Tensor | None = None):
    B, _, H, W = pointclouds.xyz.shape

    # top, left = H // 4, W // 4
    # size = H - (top*2)
    # xyz_roi = pointclouds.xyz[:, :, top:top+size, left:left+size]

    xyz_roi = pointclouds.xyz

    # downscale feature map to speed up computation and avoid single pixel overfitting
    new_size = 80
    xyz_roi = F.interpolate(xyz_roi, (new_size, new_size), mode='bilinear', align_corners=False)

    verts = xyz_roi.reshape(len(pointclouds), 3, -1).permute(0, 2, 1)  # [N, 3, H, W] -> [N, V, 3]

    if occluded is None:
        distance = chamfer_distance(landmarks3d.float(), verts, single_directional=True)[0]
    else:
        distance = torch.zeros(1, requires_grad=True, device=pointclouds.device)
        for lms, v, occ in zip(landmarks3d.float(), verts, occluded):
            distance = distance + chamfer_distance(
                lms[~occ].unsqueeze(0),
                v.unsqueeze(0),
                single_directional=True)[0] * (1.0 / B)

    return distance


def symmetry_loss(feature_maps: torch.Tensor, gpc_params: GPCParams):

    flipped = torch.flip(feature_maps, dims=[3])

    if not gpc_params.polar:
        # invert x-axis channels
        flipped[:, 0] *= -1

    def slice_to_list(s: slice):
        return list(range(s.stop)[s])

    channels = [
            slice_to_list(gpc_params.coord_channels),
            slice_to_list(gpc_params.opacity_channels),
            # only consider rgb channels from spherical harmonics
            slice_to_list(gpc_params.scale_channels),
            # slice_to_list(gpc_params.shs_channels)[:3],
    ]
    channels = list(itertools.chain(*channels))

    loss = F.mse_loss(flipped[:, channels], feature_maps[:, channels])

    if gpc_params.polar:
        loss = loss +  F.mse_loss(flipped[:, :1], feature_maps[:, :1]) * 10
    else:
        loss = loss +  F.mse_loss(flipped[:, :3], feature_maps[:, :3]) * 50

    rgb_channels = slice_to_list(gpc_params.shs_channels)[:3],
    loss = loss +  F.mse_loss(flipped[:, rgb_channels], feature_maps[:, rgb_channels]) * 0.05

    # print(loss.item())
    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(2)
    # axes[0].imshow(to_numpy(feature_maps[0, 2]))
    # axes[1].imshow(to_numpy(flipped[0, 2]))
    # plt.show()
    return loss

# import time

def create_face_weights(
        target_image: torch.Tensor,
        keypoints: np.ndarray,
        keypoint_ids: list[int],
        keypoint_ids_iris: list[int] | None = None,
        radius: int = 3,
        w_face=1.0,
        w_iris=1.0
) -> torch.Tensor:

    C, H, W = target_image.shape
    radius = int(H/168.0 * radius)

    weights = np.zeros((H, W), dtype=np.float32)

    for pt in keypoints[keypoint_ids]:
        cv2.circle(weights, pt.astype(int), radius, color=w_face, thickness=-1, lineType=cv2.LINE_AA)

    # kernel = np.ones((5, 5), np.uint8)
    # for i in range(len(weights)):
    #     weights[i] = cv2.dilate(weights[i], kernel=kernel, iterations=1)
    #     weights[i] = cv2.erode(weights[i], kernel=kernel, iterations=1)

    if keypoint_ids_iris is not None:
        radius_iris = int(H / 168.0 * (radius * 0.5))
        for pt in keypoints[keypoint_ids_iris]:
            cv2.circle(weights, pt.astype(int), radius_iris, color=w_iris, thickness=-1, lineType=cv2.LINE_AA)

    # radius_forehead = int(radius*2.0)
    # for pt in keypoints[KP_IDS_FOREHEAD]:
    #     cv2.circle(weights, pt.astype(int), radius_forehead, color=1.0, thickness=-1, lineType=cv2.LINE_AA)

    # import matplotlib.pyplot as plt
    # plt.imshow(weights)
    # plt.show()

    # t = time.time()
    weights = torch.tensor(weights).unsqueeze(0).unsqueeze(0)
    weights = kornia.filters.box_blur(weights, (11, 11))
    # print("\t t tensor", time.time()-t)

    return weights[0]


def feature_repulsion_loss(features):
    """ compute squared pairwise euclidean distances"""
    features = features.reshape(features.shape[0], -1)
    dist2 = torch.cdist(features, features, p=2.0) ** 2
    return torch.exp(-2 * dist2).mean()
    # return torch.log(loss)


def regularization_loss(feature_maps: torch.Tensor, gpc_params: GPCParams, weights: RegularizationWeights) -> torch.Tensor:
    loss_xyz = (feature_maps[:, gpc_params.coord_channels] ** 2).mean()
    loss_opac = (feature_maps[:, gpc_params.opacity_channels] ** 2).mean()
    loss_scale = (feature_maps[:, gpc_params.scale_channels] ** 2).mean()
    loss_rot = (feature_maps[:, gpc_params.rotation_channels] ** 2).mean()
    # loss_shs = (feature_maps[:, gpc_params.rotation_channels] ** 2).mean()
    loss = (
            weights.xyz * loss_xyz +
            weights.opac * loss_opac +
            weights.scale * loss_scale +
            weights.rot * loss_rot
    )
    return loss



