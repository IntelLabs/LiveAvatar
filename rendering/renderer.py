from __future__ import annotations
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import math

try:
    diff_gaussian_rasterization_available = True
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
except:
    print("Warning: diff_gaussian_rasterization is not installed. Defaulting to pytorch render backend. "
          "This implementation is extremely slow!!!")
    diff_gaussian_rasterization_available = False

from pytorch3d.renderer import FoVPerspectiveCameras
from models.gaussian_pointclouds import GaussianPointclouds
from utils.util import clip_coords_to_screen_coords, homogeneous, get_fovs_from_camera


def build_rotation(r: torch.Tensor):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device=r.device)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R


def build_scaling_rotation(s: torch.Tensor, r: torch.Tensor):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device=s.device)
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L


def strip_lowerdiag(L: torch.Tensor):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device=L.device)
    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_covariance_3d(s, r):
    L = build_scaling_rotation(s, r)
    actual_covariance = L @ L.transpose(1, 2)
    return actual_covariance
    # symm = strip_symmetric(actual_covariance)
    # return symm


def fov_to_focal(fov):
    return 1.0 / math.tan(fov / 2.0)


def build_covariance_2d(mean3d: torch.Tensor, cov3d: torch.Tensor, viewmatrix, fov_x, fov_y, image_width, image_height):
    fx = fov_to_focal(fov_x) * image_width * 0.5
    fy = fov_to_focal(fov_y) * image_height * 0.5

    t = viewmatrix @ homogeneous(mean3d).T
    t = t.T

    # truncate the influences of gaussians far outside the frustum.
    tan_fovx = math.tan(fov_x * 0.5)
    tan_fovy = math.tan(fov_y * 0.5)
    tx = (t[..., 0] / t[..., 2]).clip(min=-tan_fovx * 1.3, max=tan_fovx * 1.3) * t[..., 2]
    ty = (t[..., 1] / t[..., 2]).clip(min=-tan_fovy * 1.3, max=tan_fovy * 1.3) * t[..., 2]
    tz = t[:, 2]

    # Eq.29 locally affine transform
    # perspective transform is not affine so we approximate with first-order taylor expansion
    # notice that we multiply by the intrinsic so that the variance is at the screen space
    L = mean3d.shape[0]
    J = torch.zeros((L, 3, 3), device=mean3d.device)
    J[:, 0, 0] = 1 / tz * fx
    J[:, 0, 2] = -tx / (tz * tz) * fx
    J[:, 1, 1] = 1 / tz * fy
    J[:, 1, 2] = -ty / (tz * tz) * fy
    W = viewmatrix[:3, :3]  # transpose to correct viewmatrix
    cov2d = J @ W @ cov3d @ W.T @ J

    # add low pass filter here according to E.q. 32
    # filter = torch.eye(2, 2, device=mean3d.device) * 0.3 * (1.0 / 256**2)
    # return cov2d[:, :2, :2] + filter
    return cov2d[:, :2, :2]


def get_radius(cov2d):
    det = cov2d[:, 0, 0] * cov2d[:,1,1] - cov2d[:, 0, 1] * cov2d[:,1,0]
    mid = 0.5 * (cov2d[:, 0,0] + cov2d[:,1,1])
    lambda1 = mid + torch.sqrt((mid**2-det).clip(min=0.1))
    lambda2 = mid - torch.sqrt((mid**2-det).clip(min=0.1))
    return 3.0 * torch.sqrt(torch.max(lambda1, lambda2)).ceil()


@torch.no_grad()
def get_radius(cov2d):
    """
    Compute the radius of 2D Gaussians using an efficient eigenvalue approximation.

    Args:
        cov2d: Tensor of 2D Gaussian covariance matrices, shape (P, 2, 2).

    Returns:
        A tensor of radii for each Gaussian, shape (P,).
    """
    # Determinant and trace of the covariance matrix
    det = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] * cov2d[:, 1, 0]
    trace = cov2d[:, 0, 0] + cov2d[:, 1, 1]

    # Compute eigenvalues using quadratic formula for 2x2 symmetric matrices
    sqrt_term = torch.sqrt(torch.clamp((trace / 2) ** 2 - det, min=1e-8))  # Avoid negatives
    lambda1 = trace / 2 + sqrt_term
    lambda2 = trace / 2 - sqrt_term

    # Use the larger eigenvalue to compute the radius
    max_lambda = torch.max(lambda1, lambda2)
    radii = 3.0 * torch.sqrt(max_lambda)

    return radii.ceil()


@torch.no_grad()
def get_rect(means2d: torch.Tensor, radii: torch.Tensor, width: int, height: int):
    """
    Compute the bounding rectangle of 2D Gaussians given their means and radii.

    Args:
        means2d: Tensor of 2D Gaussian centers, shape (P, 2).
        radii: Tensor of radii for each Gaussian, shape (P,).
        width: Width of the image canvas.
        height: Height of the image canvas.

    Returns:
        A tuple of two tensors:
            - rect_min: The top-left corners of the rectangles, shape (P, 2).
            - rect_max: The bottom-right corners of the rectangles, shape (P, 2).
    """
    # Compute top-left and bottom-right coordinates of the bounding rectangles
    rect_min = torch.clip(means2d - radii[:, None], min=0)
    rect_max = torch.clip(means2d + radii[:, None], max=torch.tensor([width - 1, height - 1], device=means2d.device))
    return rect_min, rect_max


def compute_inverted_covariance(covariance_2d: torch.Tensor) -> torch.Tensor:
    """
    Compute the inverse covariance matrix

    For a 2x2 matrix
    given as
    [[a, b],
     [c, d]]
     the determinant is ad - bc

    To get the inverse matrix reshuffle the terms like so
    and multiply by 1/determinant
    [[d, -b],
     [-c, a]] * (1 / determinant)
    """
    determinant = (
        covariance_2d[:, 0, 0] * covariance_2d[:, 1, 1]
        - covariance_2d[:, 0, 1] * covariance_2d[:, 1, 0]
    )
    determinant = torch.clamp(determinant, min=1e-3)
    inverse_covariance = torch.zeros_like(covariance_2d)
    inverse_covariance[:, 0, 0] = covariance_2d[:, 1, 1] / determinant
    inverse_covariance[:, 1, 1] = covariance_2d[:, 0, 0] / determinant
    inverse_covariance[:, 0, 1] = -covariance_2d[:, 0, 1] / determinant
    inverse_covariance[:, 1, 0] = -covariance_2d[:, 1, 0] / determinant
    return inverse_covariance


class GaussRenderer(nn.Module):
    def __init__(self, image_width: int, image_height: int,
                 background_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
                 render_backend: str = 'inria'):
        super(GaussRenderer, self).__init__()
        self.image_width = image_width
        self.image_height = image_height
        self.background_color = background_color
        self.render_backend = render_backend
        if self.render_backend == 'inria' and not diff_gaussian_rasterization_available:
            self.render_backend = 'pytorch'

        self.bg_color = torch.tensor(background_color, device='cuda')

    def _project_points(
            self,
            points: torch.Tensor,
            view_matrix: torch.Tensor,
            projection_matrix: torch.Tensor
    ):
        coords = (projection_matrix @ view_matrix @ homogeneous(points).T).T
        p_w = 1.0 / (coords[..., -1:] + 0.000001)
        coords = coords * p_w
        p_view = (view_matrix @ homogeneous(points).T).T
        in_mask = p_view[..., 2] >= 0.2
        return coords, p_view, in_mask

    def _splat_gaussians2d(
            self,
            means2d: torch.Tensor,
            cov2d: torch.Tensor,
            color: torch.Tensor,
            opacity: torch.Tensor,
            depths: torch.Tensor,
    ):
        """
        Render 2D Gaussian splats onto an image using tiles for efficient processing.

        Args:
            means2d: Tensor of 2D Gaussian centers, shape (P, 2).
            cov2d: Tensor of 2D Gaussian covariance matrices, shape (P, 2, 2).
            color: Tensor of colors for each Gaussian, shape (P, 3).
            opacity: Tensor of opacity values for each Gaussian, shape (P, 1).
            depths: Tensor of depth values for each Gaussian, shape (P, 1).
            cam: Camera object providing image dimensions.
        Returns:
            A dictionary containing rendered color, depth, alpha, visibility filter, and radii.
        """

        device = means2d.device

        pix_coord = torch.stack(
            torch.meshgrid(torch.arange(self.image_width),
                           torch.arange(self.image_height),
                           indexing='xy'),
            dim=-1
        ).to(device)

        # Calculate radii and rectangular bounds for each Gaussian
        radii = get_radius(cov2d)
        rect = get_rect(means2d, radii, width=self.image_width, height=self.image_height)

        # Initialize render buffers
        img_shape = (self.image_height, self.image_width)
        render_color = torch.zeros((*img_shape, 3), device=device)
        render_depth = torch.zeros((*img_shape, 1), device=device)
        render_alpha = torch.zeros((*img_shape, 1), device=device)

        TILE_SIZE = 16
        for h_start in range(0, self.image_height, TILE_SIZE):
            for w_start in range(0, self.image_width, TILE_SIZE):
                h_end = min(h_start + TILE_SIZE, self.image_height)
                w_end = min(w_start + TILE_SIZE, self.image_width)

                # Extract tile coordinates and check Gaussian intersection
                over_tl = rect[0][..., 0].clip(min=w_start), rect[0][..., 1].clip(min=h_start)
                over_br = rect[1][..., 0].clip(max=w_end - 1), rect[1][..., 1].clip(max=h_end - 1)
                in_mask = (over_br[0] > over_tl[0]) & (over_br[1] > over_tl[1])

                if in_mask.sum() == 0:
                    continue

                # Extract relevant data for Gaussians in the current tile
                tile_coord = pix_coord[h_start:h_end, w_start:w_end].reshape(-1, 2)
                sorted_depths, index = torch.sort(depths[in_mask], descending=False)
                sorted_means2d = means2d[in_mask][index]
                sorted_cov2d = cov2d[in_mask][index]
                sorted_conic = compute_inverted_covariance(sorted_cov2d)
                sorted_opacity = opacity[in_mask][index]
                sorted_color = color[in_mask][index]

                # Compute Gaussian weights for each pixel in the tile
                dx = tile_coord[:, None, :] - sorted_means2d[None, :]
                gauss_weight = torch.exp(-0.5 * (
                        dx[..., 0] ** 2 * sorted_conic[..., 0, 0] +
                        dx[..., 1] ** 2 * sorted_conic[..., 1, 1] +
                        2 * dx[..., 0] * dx[..., 1] * sorted_conic[..., 0, 1]
                ))

                # Compute alpha and blending weights
                alpha = torch.clamp(gauss_weight[..., None] * sorted_opacity[None], max=0.99)
                T = torch.cat([torch.ones_like(alpha[:, :1]), 1 - alpha[:, :-1]], dim=1).cumprod(dim=1)
                acc_alpha = (alpha * T).sum(dim=1)

                # Compute tile color and depth
                white_bkgd = False
                tile_color = (T * alpha * sorted_color).sum(dim=1) + (1 - acc_alpha) * (1 if white_bkgd else 0)
                tile_depth = ((T * alpha) * sorted_depths[None, :, None]).sum(dim=1)

                # Update render buffers
                render_color[h_start:h_end, w_start:w_end] = tile_color.reshape(h_end - h_start, w_end - w_start, -1)
                render_depth[h_start:h_end, w_start:w_end] = tile_depth.reshape(h_end - h_start, w_end - w_start, -1)
                render_alpha[h_start:h_end, w_start:w_end] = acc_alpha.reshape(h_end - h_start, w_end - w_start, -1)

        return {
            "render": render_color,
            "depth": render_depth,
            "alpha": render_alpha,
            "visiility_filter": radii > 0,
            "radii": radii
        }

    def render(self,
               means3d: torch.Tensor,
               scales: torch.Tensor,
               rotations: torch.Tensor,
               colors: torch.Tensor,
               opacities: torch.Tensor,
               cam: FoVPerspectiveCameras,
               sh_degree: int
               ) -> dict:

        assert len(means3d.shape) == 2

        viewmatrix = cam.get_world_to_view_transform().get_matrix()[0].T
        projection_matrix = cam.get_projection_transform().get_matrix()[0].T
        fov_x, fov_y = get_fovs_from_camera(cam)

        cov3d = build_covariance_3d(scales, rotations)
        cov2d = build_covariance_2d(
            mean3d=means3d,
            cov3d=cov3d,
            viewmatrix=viewmatrix,
            fov_x=fov_x,
            fov_y=fov_y,
            image_width=self.image_width,
            image_height=self.image_height,
        )
        assert len(cov3d.shape) == 3

        means_ndc, means_view, in_mask = self._project_points(
            points=means3d,
            view_matrix=viewmatrix,
            projection_matrix=projection_matrix
        )
        # means_ndc = means_ndc[in_mask]
        # means_view = means_view[in_mask]
        depths = means_view[:, 2]
        means2d = clip_coords_to_screen_coords(means_ndc, display_size=(self.image_width, self.image_height))

        # scale covariances by image size
        # cov_scale_matrix = torch.tensor(
        #     [[self.image_width, 0],
        #      [0, self.image_height]],
        #     dtype=torch.float32,
        #     device=cov2d.device
        # )
        # cov2d = self.cov_scale_matrix @ cov2d @ self.cov_scale_matrix.T

        renders = self._splat_gaussians2d(
            means2d=means2d,
            cov2d=cov2d,
            color=colors,
            opacity=opacities,
            depths=depths,
        )
        return renders

    def render_inria(
            self,
            means3d: torch.Tensor,
            scales: torch.Tensor,
            rotations: torch.Tensor,
            colors: torch.Tensor,
            opacities: torch.Tensor,
            cam: FoVPerspectiveCameras,
            sh_degree: int,
            camera_center,
            background_color=(0,0,0),
            viewmatrix=None,
            full_projection_matrix=None,
    ) -> torch.Tensor:

        device = "cuda"
        means3d = means3d.to(device)
        scales = scales.to(device)
        rotations = rotations.to(device)
        colors = colors.to(device)
        opacities = opacities.to(device)
        cam = cam.to(device)

        if viewmatrix is None:
            viewmatrix = cam.get_world_to_view_transform().get_matrix()[0]

        if full_projection_matrix is None:
            full_projection_matrix = cam.get_full_projection_transform().get_matrix()[0]

        device = means3d.device

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(means3d, dtype=means3d.dtype, requires_grad=True, device=device)
        try:
            screenspace_points.retain_grad()
        except:
            pass

        fov_x, fov_y = get_fovs_from_camera(cam)

        # Set up rasterization configuration
        tanfovx = math.tan(fov_x * 0.5)
        tanfovy = math.tan(fov_y * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(self.image_height),
            image_width=int(self.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.bg_color,
            scale_modifier=1.0,
            viewmatrix=viewmatrix,
            projmatrix=full_projection_matrix,
            sh_degree=sh_degree,
            campos=camera_center,
            prefiltered=False,
            debug=True,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means2D = screenspace_points

        results = rasterizer(
            means3D=means3d,
            means2D=means2D,
            shs=None,
            colors_precomp=colors,
            opacities=opacities,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None
        )
        return torch.flip(results[0], dims=[1, 2]).clamp(0, 1.0)

    def forward(
            self,
            model: GaussianPointclouds,
            camera: FoVPerspectiveCameras,
            use_colors: bool = True,
            background_color=(0.,0.,0.),
            camera_center=None,
            viewmatrix=None,
            full_projection_matrix=None,
    ) -> torch.Tensor:
        means3d: torch.Tensor = model.get_xyz()
        opacities: torch.Tensor = model.get_opacity()
        scales: torch.Tensor = model.get_scaling()
        rotations: torch.Tensor = model.get_rotation()
        #     camera_center = camera.get_camera_center()[0]
        colors = model.get_colors(camera_center)
        sh_degree = model.sh_degree

        if self.render_backend == 'inria':
            return self.render_inria(
                means3d,
                scales=scales,
                rotations=rotations,
                colors=colors,
                opacities=opacities,
                cam=camera,
                sh_degree=sh_degree,
                camera_center=camera_center,
                background_color=background_color,
                viewmatrix=viewmatrix,
                full_projection_matrix=full_projection_matrix
            )
        elif self.render_backend == 'pytorch':
            renders = self.render(
                means3d,
                scales=scales,
                rotations=rotations,
                colors=colors,
                opacities=opacities,
                cam=camera,
                sh_degree=sh_degree,
            )
            return renders['render'].permute(2, 0, 1)
        else:
            raise ValueError(f"Unknown render backend {self.render_backend}")

    def render_images(
            self,
            pointclouds: GaussianPointclouds,
            cameras: FoVPerspectiveCameras | list[FoVPerspectiveCameras],
            # background_color = (1., 1., 1.)
    ) -> torch.Tensor:
        N = len(pointclouds)
        h, w = self.image_height, self.image_width
        images = torch.zeros(N, 3, h, w, dtype=torch.float32, device=pointclouds.device)
        camera_centers = cameras.get_camera_center()
        viewmatrices = cameras.get_world_to_view_transform().get_matrix()
        full_projection_matrix = cameras.get_full_projection_transform().get_matrix()
        for img_id in range(N):
            if len(cameras) == 1:
                images[img_id] = self.forward(pointclouds[img_id],
                                              cameras,
                                              background_color=self.background_color,
                                              camera_center=camera_centers[0])
            else:
                images[img_id] = self.forward(pointclouds[img_id],
                                              cameras[img_id],
                                              background_color=self.background_color,
                                              camera_center=camera_centers[img_id],
                                              viewmatrix=viewmatrices[img_id],
                                              full_projection_matrix=full_projection_matrix[img_id])
        return images
