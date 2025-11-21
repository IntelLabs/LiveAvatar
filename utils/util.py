from __future__ import annotations
import os
import numpy as np
import torch
import kornia as K
from pytorch3d.renderer import FoVPerspectiveCameras, look_at_view_transform
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix, euler_angles_to_matrix
import cv2

from utils.nn import to_numpy


def makedirs(path):
    out_dir = path
    if os.path.splitext(path)[1]:  # file
        out_dir = os.path.split(path)[0]
    if out_dir:
        try:
            os.makedirs(out_dir)
        except FileExistsError:
            pass


def homogeneous(points):
    """
    homogeneous points
    :param points: [..., 3]
    """
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def homogeneous_points(pts):
    return np.hstack([pts, np.ones((len(pts), 1))])


def homogeneous_matrix(M):
    if M is None:
        return None
    assert M.shape[0] <= 4 and M.shape[1] <= 4
    M_hom = np.eye(4)
    M_hom[:M.shape[0], :M.shape[1]] = M
    return M_hom


def project_points(points: np.ndarray,
                   model_matrix=None,
                   view_matrix=None,
                   projection_matrix=None):

    if len(points.shape) == 1:
        points = points.reshape(-1, 3)

    if model_matrix is None:
        model_matrix = np.eye(4, 4)

    if view_matrix is None:
        view_matrix = np.eye(4, 4)

    with_projection = True
    if projection_matrix is None:
        projection_matrix = np.eye(4, 4)
        with_projection = False

    points_hom = homogeneous_points(points)
    coords = projection_matrix @ view_matrix @  model_matrix @ points_hom.T
    coords = coords.T

    if with_projection:
        coords[:, 0] = coords[:, 0] / coords[:, 3]
        coords[:, 1] = coords[:, 1] / coords[:, 3]
        coords[:, 2] = coords[:, 2] / coords[:, 3]
        coords[:, 3] = 1.0
        # ndc[:, 2] = ndc[:, 2] / ndc[:, 3]

    return coords


def clip_coords_to_screen_coords__(
        normalized_coords: np.ndarray | torch.Tensor,
        display_size: tuple[int, int]
):
    if isinstance(normalized_coords, torch.Tensor):
        screen_coords = normalized_coords.clone()
    else:
        screen_coords = normalized_coords.copy()

    # flip horizontal
    invert = -1

    screen_coords[:,0] *= invert * display_size[0] / 2.0
    screen_coords[:,1] *= invert * display_size[1] / 2.0
    screen_coords[:,0] += display_size[0] / 2.0
    screen_coords[:,1] += display_size[1] / 2.0

    return screen_coords[:, :2]


def clip_coords_to_screen_coords(
        normalized_coords: np.ndarray | torch.Tensor,
        display_size: tuple[int, int]
):
    if isinstance(normalized_coords, torch.Tensor):
        screen_coords = normalized_coords.clone()
    else:
        screen_coords = normalized_coords.copy()

    # flip horizontal
    invert = -1

    screen_coords[...,0] *= invert * display_size[0] / 2.0
    screen_coords[...,1] *= invert * display_size[1] / 2.0
    screen_coords[...,0] += display_size[0] / 2.0
    screen_coords[...,1] += display_size[1] / 2.0


    out = torch.cat([screen_coords[..., 0:1].clip(0, display_size[0]-1),
                     screen_coords[..., 1:2].clip(0, display_size[1]-1)], dim=-1)
    return out
    # return screen_coords[..., :2]


def get_fovs_from_camera(cam: FoVPerspectiveCameras):
    # return np.radians(10.0), np.radians(10.0)
    fov_y = np.radians(cam.fov[0].item())
    fov_x = fov_y * cam.aspect_ratio[0].item()
    return fov_x, fov_y


def compute_camera_transform(L1, L2):
    """
    Compute the relative camera transform (R, t) between two sets of 3D landmarks.
    :param L1: Nx3 array of 3D landmarks from the first camera.
    :param L2: Nx3 array of 3D landmarks from the second camera.
    :return: R (3x3 rotation matrix), t (3x1 translation vector)
    """
    # Compute centroids
    c1 = np.mean(L1, axis=0)
    c2 = np.mean(L2, axis=0)

    # Center the points
    Q1 = L1 - c1
    Q2 = L2 - c2

    # Compute the cross-covariance matrix
    H = Q2.T @ Q1

    # SVD decomposition
    U, _, Vt = np.linalg.svd(H)
    R = U @ Vt

    # Ensure a proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        U[:, -1] = -U[:, -1]
        R = U @ Vt

    # Estimate the scaling factor
    norm1 = np.linalg.norm(Q1, axis=1)
    norm2 = np.linalg.norm(Q2, axis=1)
    scale = 1.0 / (np.sum(norm1 * norm2) / np.sum(norm2 ** 2))

    # Compute translation
    t = c2 - scale * R @ c1

    M = np.eye(4)
    M[:3, :3] = scale * R
    M[:3, 3] = t
    return M.astype(np.float32)


def convert_to_9d(cameras: FoVPerspectiveCameras):
    """
    Convert FoVPerspectiveCameras rotation matrices to 6D continuous representation.

    Args:
        cameras (FoVPerspectiveCameras): PyTorch3D camera object.

    Returns:
        torch.Tensor: 6D continuous representation of rotations, shape (batch_size, 6).
    """
    # Take the first two columns of the rotation matrix
    R_6d = matrix_to_rotation_6d(cameras.R)

    # Concatenate R_6d and T (B, 9)
    pose_9d = torch.cat([R_6d, cameras.T], dim=1)
    return pose_9d


# _R, _T = look_at_view_transform(eye=np.array([[0., 0., 5]]))
# cam_front_mid = FoVPerspectiveCameras(R=_R, T=_T, fov=20.0, device='cuda')
# pose_6d_front_mid = matrix_to_rotation_6d(cam_front_mid.R)


def reconstruct_pose_from_9d(pose_9d):
    """
    Reconstruct the rotation matrix and translation vector from a 9D pose representation.

    Args:
        pose_9d (torch.Tensor): 9D pose representation, shape (batch_size, 9).

    Returns:
        torch.Tensor: Rotation matrices, shape (batch_size, 3, 3).
        torch.Tensor: Translation vectors, shape (batch_size, 3).
    """
    # Split into 6D rotation and 3D translation
    R = rotation_6d_to_matrix(torch.tanh(pose_9d[:, :6]))  #+ pose_6d_front_mid)
    T = pose_9d[:, 6:] + cam_front_mid.T
    return R, T


def create_camera(azimuth, elevation, distance=None, fov=30.0):
    if distance is None:
        distance = 0.5 / np.tan(np.radians(fov/2))
    x = np.sin(np.radians(azimuth)) * distance
    y = np.sin(np.radians(elevation)) * distance
    z = np.cos(np.radians(azimuth)) * distance
    cam_pos =np.array([[x, y, z]])
    head_center = np.array([[0, 0, -0.25]])  # look slightly 'behind' face
    R, T = look_at_view_transform(eye=cam_pos, at=head_center)
    return FoVPerspectiveCameras(R=R, T=T, fov=fov).to('cuda')


def get_camera_azimuth(cameras):
    camera_center = to_numpy(cameras.get_camera_center())
    camera_x = camera_center[:, 0]
    camera_z = camera_center[:, 2]
    return np.degrees(np.arctan(camera_x / camera_z))


import kornia

def init_uv_maps(_n, _w, _h):
    grid2d = np.dstack(np.meshgrid(range(_w), range(_h)))
    return grid2d[np.newaxis].repeat(_n, axis=0)


def spatial_transform(X: torch.Tensor, map: torch.Tensor, mode='bilinear', replace_nan=True) -> torch.Tensor:
    if X is None:
        return None

    assert isinstance(X, torch.Tensor)
    assert isinstance(map, torch.Tensor)

    assert X.ndim == 4
    assert map.ndim == 4
    assert map.shape[3] == 2

    X_new = kornia.geometry.transform.remap(X.float(), map[..., 0], map[..., 1], align_corners=True, mode=mode)
    return X_new


def log_mapping(x, k=3.0):

    if k is None or k <= 0:
        return x

    # to range [-1, +1]
    x_min, x_max = x.min(), x.max()
    x = (x - x.min()) / (x.max() - x.min()) * 2 - 1

    # apply scaling
    x = torch.sign(x) * ((torch.exp(k * torch.abs(x)) - 1) / (np.exp(k) - 1))

    # to original range
    return (x+1)/2 * (x_max - x_min) + x_min


def remap_feature_map(feature_map):
    W, H = feature_map.shape[-2:]
    xy_map = torch.tensor(init_uv_maps(1, W, H))
    xy_map[..., 0] = log_mapping(xy_map[..., 0], 0.25)
    xy_map[..., 1] = log_mapping(xy_map[..., 1], 1.75)
    return spatial_transform(feature_map, xy_map.cuda(), mode='bilinear')


class PolarToCartesian():

    def __init__(self, theta_dims, phi_dims, k_phi=None, k_theta=None, device='cuda'):
        super().__init__()
        theta = torch.linspace(0, np.pi, theta_dims, device=device)
        theta = log_mapping(theta, k_theta)

        phi = torch.linspace(0, 2 * np.pi, phi_dims, device=device)
        phi = log_mapping(phi, k_phi)

        phi, theta = torch.meshgrid(phi, theta, indexing='xy')

        self.cos_theta = torch.cos(theta)
        self.sin_theta_cos_phi = torch.sin(theta) * torch.cos(phi)
        self.sin_theta_sin_phi = torch.sin(theta) * torch.sin(phi)

        dtype = torch.float32
        rotation_matrix_y = torch.tensor([
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0]
        ], dtype=dtype, device=device)

        rotation_matrix_z = torch.tensor([
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1]
        ], dtype=dtype, device=device)

        self._rotation_matrix = rotation_matrix_z @ -rotation_matrix_y

    def __call__(self, sht, scales):
        """ Convert to Cartesian coordinates """

        # scales = scales.reshape(-1, 3, 1, 1)
        # x = sht * self.sin_theta_cos_phi * scales[:, 0:1]
        # y = sht * self.sin_theta_sin_phi * scales[:, 1:2]
        # z = sht * self.cos_theta * scales[:, 2:3]
        x = sht * self.sin_theta_cos_phi
        y = sht * self.sin_theta_sin_phi
        z = sht * self.cos_theta

        # print(f"t_cartesian: {time.time()-t}")

        # Visualize the resulting shape
        # import matplotlib.pyplot as plt
        # fig = plt.figure(figsize=(8, 8))
        # ax = fig.add_subplot(111, projection='3d')
        # # ax.plot_surface(to_numpy(x.reshape(1, -1)).T, to_numpy(y.reshape(1, -1)).T, to_numpy(z.reshape(1, -1)).T,
        # #                 rstride=1, cstride=1, color='b', alpha=0.7, edgecolor='k')
        # ax.scatter(to_numpy(x[0,0].ravel()), to_numpy(y[0,0].ravel()), to_numpy(z[0, 0].ravel()))
        # ax.set_box_aspect([1, 1, 1])
        # plt.show()

        rotated_coords = self._rotation_matrix @ torch.stack([x.ravel(), y.ravel(), z.ravel()])
        x, y, z = rotated_coords.reshape(3, *x.shape)
        # x, y, z = z, -x, y
        xyz = torch.cat([x, y, z], dim=1)
        return xyz


def transform_camera(camera: FoVPerspectiveCameras, M: torch.Tensor, fov=None):
    cam_pos = camera.get_camera_center()

    R = camera.R
    view_direction = R[:, 2, :]  # Camera's forward direction (third row of R)
    look_at = cam_pos + view_direction  # New look-at target

    # Transform camera position and look-at target
    cam_pos_new = (M @ homogeneous(cam_pos).T)[:, :3, 0]
    look_at_new = (M @ homogeneous(look_at).T)[:, :3, 0]

    # Compute transformed up vector
    up_vec = torch.tensor([[0, 1, 0]]).to(M.device) + cam_pos  # Define an "up" direction
    up_vec_new = (M @ homogeneous(up_vec).T)[:, :3, 0]
    up_new = up_vec_new - cam_pos_new
    up_new = up_new / torch.norm(up_new)

    R_new, T_new = look_at_view_transform(eye=cam_pos_new, at=look_at_new, up=up_new, device=M.device)
    if fov is None:
        fov = camera.fov
    cam_new = FoVPerspectiveCameras(R=R_new, T=T_new, fov=fov, device=M.device)
    return cam_new


def transform_landmarks(lms, M):
    return (M @ homogeneous_points(lms).T).T[:, :3]


def transform_landmarks_torch(lms: torch.Tensor, M: torch.Tensor):
    return (M @ homogeneous(lms).T).transpose(1, 2)[:, :, :3]


def stack_cameras(camera_list: list[FoVPerspectiveCameras]) -> FoVPerspectiveCameras:
    R = torch.cat([cam.R for cam in camera_list], dim=0)
    T = torch.cat([cam.T for cam in camera_list], dim=0)
    fov = torch.cat([cam.fov for cam in camera_list])
    return FoVPerspectiveCameras(R=R, T=T, fov=fov, device=camera_list[0].device)


def backproject_landmarks(landmarks, cam, width, height):
    if not isinstance(landmarks, torch.Tensor):
        landmarks = torch.tensor(landmarks, dtype=torch.float32, device=cam.device)
    clip_coords = cam.transform_points_ndc(landmarks)
    if len(clip_coords.shape) == 3 and clip_coords.shape[0] == 1:
        clip_coords = clip_coords[0]
    screen_coords = clip_coords_to_screen_coords(clip_coords, (width, height))
    return screen_coords


def pairwise_backproject_landmarks(
        landmarks: torch.Tensor, cameras: FoVPerspectiveCameras, width: int, height: int
) -> torch.Tensor:
    N = len(cameras)
    lms_repeated = landmarks.repeat(N, 1, 1)
    cams_interleaved = FoVPerspectiveCameras(
        R=cameras.R.repeat_interleave(N, dim=0),
        T=cameras.T.repeat_interleave(N, dim=0),
        fov=cameras.fov.repeat_interleave(N, dim=0),
        device=cameras.device
    )
    clip_coords = cams_interleaved.transform_points_ndc(lms_repeated)
    screen_coords = clip_coords_to_screen_coords(clip_coords, (width, height))
    return screen_coords




def pad_if_needed(
        image: np.ndarray,
        min_height: int = None,
        min_width: int = None,
        pad_value: int = 0,
        center_pad: bool = True
) -> np.ndarray:
    """
    Pads the input image if its height or width is below the specified minimums.

    Args:
        image (np.ndarray): Input image (H, W, C) or (H, W).
        min_height (int): Minimum height after padding.
        min_width (int): Minimum width after padding.
        pad_value (int or tuple): Value to use for padding.
        center_pad (bool): If True, pads equally on both sides. Else pads only at bottom/right.

    Returns:
        np.ndarray: Padded image.
    """
    h, w = image.shape[:2]

    pad_top = pad_bottom = pad_left = pad_right = 0

    if min_height is not None and h < min_height:
        pad_total = min_height - h
        if center_pad:
            pad_top = pad_total // 2
            pad_bottom = pad_total - pad_top
        else:
            pad_bottom = pad_total

    if min_width is not None and w < min_width:
        pad_total = min_width - w
        if center_pad:
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
        else:
            pad_right = pad_total

    padded = cv2.copyMakeBorder(
        image,
        top=pad_top,
        bottom=pad_bottom,
        left=pad_left,
        right=pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=pad_value
    )
    return padded


def pad_to_square(image: np.ndarray, pad_value=0, center_pad=True) -> np.ndarray:
    """
    Pads a NumPy image to make it square, using the longer side as target size.

    Args:
        image (np.ndarray): Input image (H, W, C) or (H, W).
        pad_value (int or tuple): Value for padding (e.g., 0 or (0,0,0)).
        center_pad (bool): If True, pads equally on both sides. Else pads only bottom/right.

    Returns:
        np.ndarray: Square-padded image.
    """
    h, w = image.shape[:2]
    size = max(h, w)

    pad_top = pad_bottom = pad_left = pad_right = 0

    if h < size:
        pad_total = size - h
        if center_pad:
            pad_top = pad_total // 2
            pad_bottom = pad_total - pad_top
        else:
            pad_bottom = pad_total

    if w < size:
        pad_total = size - w
        if center_pad:
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
        else:
            pad_right = pad_total

    padded = cv2.copyMakeBorder(
        image,
        top=pad_top,
        bottom=pad_bottom,
        left=pad_left,
        right=pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=pad_value
    )
    return padded


def find_images_in_dir(path, image_exts=('.png', '.jpg')):
    from pims.utils.sort import natural_keys
    import itertools
    import glob
    return itertools.chain(*[
        sorted(glob.glob(os.path.join(path, '*' + e)), key=natural_keys) for e in image_exts
    ])


def get_images(input_path):
    import pims

    valid_image_exts = ['.png', '.jpg']
    # valid_video_exts = ['.' + e for e in pims.PyAVVideoReader.class_exts()]
    valid_video_exts = ['.avi', '.mpg', '.mpeg', '.mp4']

    def is_video(path: str):
        return os.path.splitext(path)[1] in valid_video_exts

    def is_single_image(path: str):
        return os.path.splitext(path)[1] in valid_image_exts

    frame_rate = 25

    if is_video(input_path):
        source_images = pims.PyAVVideoReader(input_path)
        frame_rate = source_images.frame_rate
    elif is_single_image(input_path):
        source_images = [cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2RGB)]
    elif os.path.isdir(input_path):
        img_paths = find_images_in_dir(input_path)
        source_images = pims.ImageSequence(img_paths)
        # source_images = pims.ImageSequence(os.path.join(input_path, '*.png'))
    else:
        raise ValueError(f"Invalid input path: {input_path}")

    # if isinstance(source_images, np.ndarray) and len(source_images.shape) == 3:
    #     source_images = [source_images]

    return source_images, frame_rate



def _as_batched_M(M: torch.Tensor, B: int, device, dtype):
    """
    Ensure M has shape (B, 4, 4). Accepts (4,4) or (B,4,4).
    """
    M = M.to(device=device, dtype=dtype)
    if M.dim() == 2:
        M = M.unsqueeze(0).expand(B, -1, -1)
    return M

def _apply_M(M: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """
    Apply homogeneous transform(s) M to pts.
    M: (B,4,4)
    pts: (B,3)
    returns: (B,3)
    """
    B = pts.shape[0]
    ones = torch.ones(B, 1, device=pts.device, dtype=pts.dtype)
    pts_h = torch.cat([pts, ones], dim=-1)               # (B,4)
    out  = (M @ pts_h.unsqueeze(-1)).squeeze(-1)         # (B,4)
    return out[..., :3]


def transform_camera2(camera: FoVPerspectiveCameras, M: torch.Tensor, fov=None):
    """
    Apply a world-space transform M (rotation/translation/scaling) to the camera
    while preserving its original 'up' orientation. If M is identity, the camera
    is returned unchanged.

    camera: FoVPerspectiveCameras (B,)
    M: (4,4) or (B,4,4), world-space homogeneous transform
    """
    device = camera.device
    dtype  = camera.R.dtype
    B      = camera.R.shape[0]

    # Normalize M shape/device/dtype and short-circuit identity
    M = _as_batched_M(M, B, device, dtype)
    I = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
    if torch.allclose(M, I):
        return camera  # exactly identity -> return as-is

    # Extract camera center, forward (+Z row), and up (+Y row) in world coords
    cam_pos   = camera.get_camera_center()      # (B,3)
    forward_w = camera.R[:, :, 2]               # (B,3) camera +Z in world
    up_w      = camera.R[:, :, 1]               # (B,3) camera +Y in world

    # Build look-at and an 'up point' so we can transform as points
    look_at   = cam_pos + forward_w             # (B,3)
    up_point  = cam_pos + up_w                  # (B,3)

    # Transform all three by M
    cam_pos_new  = _apply_M(M, cam_pos)
    look_at_new  = _apply_M(M, look_at)
    up_point_new = _apply_M(M, up_point)

    # Recompute up as the transformed up-point relative to new center
    up_new = up_point_new - cam_pos_new
    up_new = up_new / (up_new.norm(dim=1, keepdim=True) + 1e-9)

    # Rebuild view
    R_new, T_new = look_at_view_transform(
        eye=cam_pos_new, at=look_at_new, up=up_new, device=device
    )
    if fov is None:
        fov = camera.fov
    cam_new = FoVPerspectiveCameras(R=R_new, T=T_new, fov=fov, device=device)
    return cam_new


# def rotation_matrix_y(theta: float, device="cpu"):
#     """Homogeneous 4x4 rotation matrix around the world y-axis."""
#     c, s = np.cos(theta), np.sin(theta)
#     R = torch.tensor([
#         [c, 0, s, 0],
#         [0, 1, 0, 0],
#         [-s, 0, c, 0],
#         [0, 0, 0, 1],
#     ], dtype=torch.float32, device=device)
#     return R

def rotation_matrix_y(theta: float, origin=(0.0, 0.0, 0.0), device="cpu", dtype=torch.float32):
    """
    Homogeneous 4x4 rotation matrix around the world y-axis,
    about a given point 'origin' (x,y,z).

    theta : float (radians) or torch scalar
    origin : tuple/list/torch tensor of shape (3,), the pivot point
    """
    if not torch.is_tensor(theta):
        theta = torch.tensor(theta, device=device, dtype=dtype)

    c, s = torch.cos(theta), torch.sin(theta)

    # Basic rotation around world y-axis at origin
    R = torch.tensor([
        [c, 0, s, 0],
        [0, 1, 0, 0],
        [-s, 0, c, 0],
        [0, 0, 0, 1],
    ], dtype=dtype, device=device)

    # Translation matrices
    if not torch.is_tensor(origin):
        origin = torch.tensor(origin, dtype=dtype, device=device)
    ox, oy, oz = origin.tolist()

    T_neg = torch.tensor([
        [1, 0, 0, -ox],
        [0, 1, 0, -oy],
        [0, 0, 1, -oz],
        [0, 0, 0, 1],
    ], dtype=dtype, device=device)

    T_pos = torch.tensor([
        [1, 0, 0, ox],
        [0, 1, 0, oy],
        [0, 0, 1, oz],
        [0, 0, 0, 1],
    ], dtype=dtype, device=device)

    # Compose: translate to origin -> rotate -> translate back
    M = T_pos @ R @ T_neg
    return M


def rotation_matrix_x(theta: float, origin=(0.0, 0.0, 0.0), device="cpu", dtype=torch.float32):
    """
    Homogeneous 4x4 rotation matrix around the world x-axis,
    about a given point 'origin' (x,y,z).
    """
    if not torch.is_tensor(theta):
        theta = torch.tensor(theta, device=device, dtype=dtype)

    c, s = torch.cos(theta), torch.sin(theta)

    R = torch.tensor([
        [1, 0,  0, 0],
        [0, c, -s, 0],
        [0, s,  c, 0],
        [0, 0,  0, 1],
    ], dtype=dtype, device=device)

    if not torch.is_tensor(origin):
        origin = torch.tensor(origin, dtype=dtype, device=device)
    ox, oy, oz = origin.tolist()

    T_neg = torch.tensor([
        [1,0,0,-ox],
        [0,1,0,-oy],
        [0,0,1,-oz],
        [0,0,0,  1],
    ], dtype=dtype, device=device)

    T_pos = torch.tensor([
        [1,0,0,ox],
        [0,1,0,oy],
        [0,0,1,oz],
        [0,0,0, 1],
    ], dtype=dtype, device=device)

    return T_pos @ R @ T_neg


def rotation_matrix(rx=0.0, ry=0.0, rz=0.0, origin=(0,0,0),
                    device="cpu", dtype=torch.float32, order="YXZ"):
    """
    Build a homogeneous 4x4 rotation matrix around world X/Y/Z
    using PyTorch3D's euler_angles_to_matrix.
    rx, ry, rz in radians.
    order = "XYZ" means: apply X, then Y, then Z.
    """
    if not torch.is_tensor(rx):
        rx = torch.tensor(rx, device=device, dtype=dtype)
    if not torch.is_tensor(ry):
        ry = torch.tensor(ry, device=device, dtype=dtype)
    if not torch.is_tensor(rz):
        rz = torch.tensor(rz, device=device, dtype=dtype)

    angles = torch.stack([rx, ry, rz]).unsqueeze(0)  # (1,3)
    R = euler_angles_to_matrix(angles, convention=order)[0]  # (3,3)

    # Homogeneous 4x4
    M = torch.eye(4, dtype=dtype, device=device)
    M[:3,:3] = R

    if not torch.is_tensor(origin):
        origin = torch.tensor(origin, dtype=dtype, device=device)
    T_neg = torch.eye(4, dtype=dtype, device=device); T_neg[:3,3] = -origin
    T_pos = torch.eye(4, dtype=dtype, device=device); T_pos[:3,3] = origin

    return T_pos @ M @ T_neg


def orbit_camera(camera, azim=0.0, elev=0.0, roll=0.0, origin=(0,0,0), fov=None, degrees=True):
    """
    Orbit the camera around 'origin' by delta_azim (yaw around world Y)
    and delta_elev (pitch around world X). Both in radians.
    """
    device = camera.device
    dtype  = camera.R.dtype

    if degrees:
        azim = np.radians(azim)
        elev = np.radians(elev)
        roll = np.radians(roll)

    M = rotation_matrix(azim, elev, roll, origin=origin, device=device, dtype=dtype)

    return transform_camera2(camera, M, fov=fov)


def crop_and_resize_from_keypoints(images, keypoints, output_size=(512, 512), padding=0.1):
    """
    Crop and resize images to a fixed size based on keypoints.

    Args:
        images: torch.Tensor of shape [B, C, H, W]
        keypoints: torch.Tensor of shape [B, K, 2]
        output_size: tuple (height, width) for output images
        padding: padding ratio around bounding box

    Returns:
        cropped_resized: torch.Tensor of shape [B, C, output_size[0], output_size[1]]
    """
    B, C, H, W = images.shape

    # Calculate bounding boxes
    min_coords = keypoints.min(dim=1)[0]
    max_coords = keypoints.max(dim=1)[0]

    x1, y1 = min_coords[:, 0], min_coords[:, 1]
    x2, y2 = max_coords[:, 0], max_coords[:, 1]

    # Add padding
    width = x2 - x1
    height = y2 - y1
    pad_x = width * padding
    pad_y = height * padding

    x1 = torch.clamp(x1 - pad_x, 0, W)
    y1 = torch.clamp(y1 - pad_y, 0, H)
    x2 = torch.clamp(x2 + pad_x, 0, W)
    y2 = torch.clamp(y2 + pad_y, 0, H)

    boxes = torch.stack([
        torch.stack([x1, y1], dim=1),
        torch.stack([x2, y1], dim=1),
        torch.stack([x2, y2], dim=1),
        torch.stack([x1, y2], dim=1),
    ], dim=1)  # [B, 4, 2]

    # Crop and resize to fixed size
    cropped_resized = K.geometry.transform.crop_and_resize(
        images,
        boxes,
        output_size,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )

    return cropped_resized