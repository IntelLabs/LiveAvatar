from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras
import numpy as np
import torch

from tracking.landmarks import convert_landmarks_mediapipe_to_dlib
from utils.util import compute_camera_transform, homogeneous_points, transform_camera
from utils.nn import to_numpy
from tracking.landmarks import lm478_to_lm68


def normalize_mesh(mesh):
    verts = mesh.verts_packed()
    N = verts.shape[0]
    center = verts.mean(0)
    scale = max((verts - center).abs().max(0)[0])
    mesh.offset_verts_(-(center.expand(N, 3)))
    mesh.scale_verts_((1.0 / float(scale)))
    return mesh


class CanonicalFaceAlignment():
    def __init__(self, canonical_obj_file, cam_pos, fov, look_at=None, only_rigid_lms=True):
        mesh = load_objs_as_meshes([canonical_obj_file], device='cpu')
        mesh = normalize_mesh(mesh)

        self._only_rigid_lms = only_rigid_lms
        self._cam_pos = cam_pos
        self._lms_ref = to_numpy(mesh.verts_packed()) * 0.25
        if only_rigid_lms:
            self._lms_ref = lm478_to_lm68(self._lms_ref)
        # self._pc_ref = Pointcloud(points=self._lms_ref, colors=GREEN)

        self._look_at = look_at
        if self._look_at is None:
            self._look_at = [[0, 0, 0]]

        self._fov = fov
        R, T = look_at_view_transform(eye=self._cam_pos, at=self._look_at)
        self._cam = FoVPerspectiveCameras(R=R, T=T, fov=self._fov)

    def update_cam_pos(self, cam_pos):
        self._cam_pos = cam_pos
        R, T = look_at_view_transform(eye=self._cam_pos, at=self._look_at)
        self._cam = FoVPerspectiveCameras(R=R, T=T, fov=self._fov)

    # def set_target(self, target_lms):
    #     self._lms_ref = target_lms

    def align(self, lms) -> (FoVPerspectiveCameras, np.ndarray):

        _lms_opt = lms.copy()

        if self._only_rigid_lms:
            if len(_lms_opt) == 478:
                _lms_opt = convert_landmarks_mediapipe_to_dlib(_lms_opt)

        M_transform = compute_camera_transform(self._lms_ref, _lms_opt)
        M_inv = np.linalg.inv(M_transform)

        lms_new = (M_inv @ homogeneous_points(lms).T).T[:, :3]
        cam_new = transform_camera(self._cam, torch.tensor(M_inv).unsqueeze(0))

        return cam_new, lms_new, M_inv
