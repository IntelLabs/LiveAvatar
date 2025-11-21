from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2
import torch
import numpy as np
import torch.optim

from utils.util import backproject_landmarks
from utils.nn import to_numpy
from matplotlib.colors import hsv_to_rgb


def _to_img_8U(img, f, dsize, interpolation):
    if img is None:
        return
    if isinstance(img, torch.Tensor):
        img = to_numpy(img.permute(1, 2, 0))
    if dsize is not None:
        img = cv2.resize(img, dsize=dsize, interpolation=interpolation)
    elif f is not None:
        img = cv2.resize(img, dsize=None, fx=f, fy=f, interpolation=interpolation)
    if img.dtype == np.float32:
        img = (img * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def show_image(
        title: str,
        img,
        f: float | None = None,
        dsize: tuple[int, int] = None,
        wait: int | None = None,
        interpolation=cv2.INTER_CUBIC
):
    img = _to_img_8U(img, f, dsize, interpolation)
    cv2.imshow(title, img)
    if wait is not None:
        cv2.waitKey(wait)


def write_image(
        filepath,
        img,
        f: float | None = None,
        dsize: tuple[int, int] = None,
        interpolation=cv2.INTER_CUBIC
):
    img = _to_img_8U(img, f, dsize, interpolation)
    cv2.imwrite(filepath, img)


def draw_channels(tensor: torch.Tensor, vmax=None, normalize=False):

    # Compute magnitude
    magnitude = torch.linalg.norm(tensor, dim=0)

    # Normalize x, y, z to unit vectors for direction
    normalized_tensor = tensor / (magnitude + 1e-8)  # Avoid division by zero
    x, y, z = normalized_tensor[0], normalized_tensor[1], normalized_tensor[2]

    # Compute spherical coordinates
    # Azimuthal angle (phi) for x-y plane
    phi = torch.atan2(y, x) / (2 * np.pi) + 0.5  # Normalize atan2 to [0, 1]

    # Polar angle (theta) for z-component
    theta = torch.acos(z.clamp(-1, 1)) / np.pi  # Normalize acos to [0, 1]

    # Map to HSV
    hue = phi  # Hue represents azimuth
    saturation = 1 - theta  # Saturation represents elevation (0 = pole, 1 = equator)
    if normalize:
        vmax = magnitude.max()
    elif vmax is None:
        vmax = 1.0
    # value = magnitude / vmax
    value = torch.clamp_max_(magnitude, vmax) / vmax

    # Stack HSV channels
    hsv_image = to_numpy(torch.stack((hue, saturation, value), dim=-1))

    # Convert HSV to RGB
    rgb_image = hsv_to_rgb(hsv_image)
    return rgb_image


def show_xyz(tensor: torch.Tensor, winname: str = "XYZ", vmax=None, normalize=False):
    rgb_image = draw_channels(tensor, vmax, normalize)
    rgb_image = cv2.resize(rgb_image, dsize=(512, 512))
    cv2.imshow(winname, cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))


def draw_channel(tensor: torch.Tensor, vmin=None, vmax=None, cmap: str | plt.Colormap | None = plt.cm.viridis):
    """
    Convert single channel feature map to colomapped RGB image
    Args:
        tensor: feature map of shape (1, H, W) or (H, W)

    Returns:
        RGB image as float numpy array of shape (3, H, W)
    """
    if len(tensor.shape) == 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return color_map(tensor, vmin=vmin, vmax=vmax, cmap=cmap).astype(np.float32)


def draw_colors(tensor: torch.Tensor):
    return to_numpy(tensor.permute(1, 2, 0))


def show_channel(tensor: torch.Tensor, winname: str = "XYZ", vmax=None, normalize=False):
    color_mapped = draw_channel(tensor, vmax=vmax)
    color_mapped = cv2.resize(color_mapped, dsize=(512, 512), interpolation=cv2.INTER_NEAREST)
    cv2.imshow(winname, cv2.cvtColor(color_mapped, cv2.COLOR_BGR2RGB))


def show_colors(tensor: torch.Tensor, winname: str = "Colors"):
    rgb_image = draw_colors(tensor)
    rgb_image = cv2.resize(rgb_image, dsize=(512, 512))
    cv2.imshow(winname, cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))


def to_silhouette(images: torch.Tensor, threshold=0.05) -> torch.Tensor:
    return (images.mean(dim=1) > threshold).float()


def cvt32FtoU8(img):
    return (img * 255.0).astype(np.uint8)


def to_disp_image(img, denorm=True, output_dtype=np.uint8):
    if not isinstance(img, np.ndarray):
        img = img.detach().cpu().numpy()
    img = img.astype(np.float32).copy()
    if img.shape[0] == 3:
        img = img.transpose((1, 2, 0)).copy()
    if img.max() > 2.00:
        if isinstance(img, np.ndarray):
            img /= 255.0
        else:
            raise ValueError("Image data in wrong value range (min/max={:.2f}/{:.2f}).".format(img.min(), img.max()))
    img = np.clip(img, a_min=0, a_max=1)
    if output_dtype == np.uint8:
        img = cvt32FtoU8(img)
    if len(img.shape) == 3 and img.shape[0] == 1:
        img = img[0]
    return img


def to_disp_images(images, denorm=False, output_dtype=np.uint8):
    return [to_disp_image(i, denorm, output_dtype) for i in images]


def color_map(data, vmin=None, vmax=None, cmap: str | plt.Colormap | None = plt.cm.viridis):
    if isinstance(cmap, str):
        cmap = mpl.colormaps[cmap]
    data = to_numpy(data)
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    val = np.maximum(vmin, np.minimum(vmax, data))
    norm = (val-vmin)/(vmax-vmin)
    cm = cmap(norm)
    if isinstance(cm, tuple):
        return cm[:3]
    if len(cm.shape) > 2:
        cm = cm[:,:,:3]
    return cm


def to_cmap_images(tensor, vmin=0, vmax=1.0, cmap: str | plt.Colormap | None = plt.cm.viridis):
    cmapped = [color_map(img, vmin=vmin, vmax=vmax, cmap=cmap) for img in to_numpy(tensor)]
    return [(img * 255).astype(np.uint8) for img in cmapped]


def to_image(
        data: np.ndarray | torch.Tensor,
        normalize: bool = False,
        vmin: float | None = None,
        vmax: float | None = None,
        cmap=plt.cm.viridis
) -> np.ndarray | None:

    if data is None or len(data) == 0:
        return None

    if isinstance(data, torch.Tensor):
        data = to_numpy(data)

    data = np.array(data)

    # make sure we have a numpy array of shape (H, W, C)

    if len(data.shape) == 4:
        data = data[0]

    if len(data.shape) == 2:
        data = data[..., np.newaxis]

    if data.shape[0] in [1, 3]:
        data = data.transpose((1, 2, 0))

    assert len(data.shape) == 3
    assert data.shape[2] in [1, 3]

    # apply color map image if it's a single channel image
    if data.shape[2] == 1:
        data = np.array(color_map(data[..., 0].astype(np.float32), vmin, vmax, cmap))

    if data.dtype == np.float64:
        data = data.astype(np.float32)

    if normalize:
        data -= data.min()
        data /= data.max()
    else:
        data[data < 0] = 0
    #     data[data > 1] = 1

    if data.dtype == np.float32:
        if data.max() < 1.01:
            data *= 255.0
        data = np.clip(data, a_min=0, a_max=255)
        data = data.astype(np.uint8)

    return data # return type np.uint8


# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid
def make_grid(
        data: list | tuple | np.ndarray | torch.tensor,
        padsize: int = 2,
        padval: int = 255,
        nrows: int | None = None,
        ncols: int | None = None,
        dsize: tuple[int, int] | None = None,
        fx: float | None = None,
        fy: float | None =None,
        normalize: bool = False,
        vmin: float = None,
        vmax: float = None,
        cmap=plt.cm.viridis,
        interpolation=cv2.INTER_LINEAR
):

    assert isinstance(data, (list, tuple, np.ndarray, torch.Tensor))

    data = np.array(
        [to_image(item, normalize=normalize, vmin=vmin, vmax=vmax, cmap=cmap) for item in data]
    )

    assert len(data.shape) == 4

    if nrows is None and ncols is None:
        nrows = 1
        # force the number of tiles to be square
        # n = int(np.ceil(np.sqrt(data.shape[0])))
        # nrows, ncols = n, n

    if ncols is None:
        ncols = int(np.ceil(data.shape[0]/float(nrows)))

    if nrows is None:
        nrows = int(np.ceil(data.shape[0]/float(ncols)))

    c = ncols
    r = nrows

    padding = ((0, r*c - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((r, c) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((r * data.shape[1], c * data.shape[3]) + data.shape[4:])

    if dsize is not None or fx is not None or fy is not None:
        data = cv2.resize(data, dsize=dsize, fx=fx, fy=fy, interpolation=interpolation)

    return data


def get_pos_in_image(loc, text_size, image_shape):
    bottom_offset = int(6*text_size)
    right_offset = int(95*text_size)
    line_height = int(35*text_size)
    mid_offset = right_offset
    top_offset = line_height + int(0.05*line_height)
    if loc == 'tl':
        pos = (2, top_offset)
    elif loc == 'tl+1':
        pos = (2, top_offset + line_height)
    elif loc == 'tl+2':
        pos = (2, top_offset + line_height * 2)
    elif loc == 'tr':
        pos = (image_shape[1]-right_offset, top_offset)
    elif loc == 'tr+1':
        pos = (image_shape[1]-right_offset, top_offset + line_height)
    elif loc == 'tr+2':
        pos = (image_shape[1]-right_offset, top_offset + line_height*2)
    elif loc == 'bl':
        pos = (2, image_shape[0]-bottom_offset)
    elif loc == 'bl-1':
        pos = (2, image_shape[0]-bottom_offset-line_height)
    elif loc == 'bl-2':
        pos = (2, image_shape[0]-bottom_offset-2*line_height)
    # elif loc == 'bm':
    #     pos = (mid_offset, image_shape[0]-bottom_offset)
    # elif loc == 'bm-1':
    #     pos = (mid_offset, image_shape[0]-bottom_offset-line_height)
    elif loc == 'br':
        pos = (image_shape[1]-right_offset, image_shape[0]-bottom_offset)
    elif loc == 'br-1':
        pos = (image_shape[1]-right_offset, image_shape[0]-bottom_offset-line_height)
    elif loc == 'br-2':
        pos = (image_shape[1]-right_offset, image_shape[0]-bottom_offset-2*line_height)
    elif loc == 'bm':
        pos = (image_shape[1]-right_offset*2, image_shape[0]-bottom_offset)
    elif loc == 'bm-1':
        pos = (image_shape[1]-right_offset*2, image_shape[0]-bottom_offset-line_height)
    elif loc == 'bm-2':
        pos = (image_shape[1]-right_offset*2, image_shape[0]-bottom_offset-2*line_height)
    else:
        raise ValueError("Unknown location {}".format(loc))
    return pos


def add_label_to_images(images, labels, gt_labels=None, loc='tl', color=(255,255,255), size=0.7, thickness=1, prefix="", suffix=""):
    new_images = to_disp_images(images)
    _labels = to_numpy(labels)
    _gt_labels = to_numpy(gt_labels)
    for idx, (disp, val) in enumerate(zip(new_images, _labels)):
        if _gt_labels is not None:
            color = (0,255,0) if _labels[idx] == _gt_labels[idx] else (255,0,0)
        # if val != 0:
        pos = get_pos_in_image(loc, size, disp.shape)
        cv2.putText(disp, prefix + str(val) + suffix, pos, cv2.FONT_HERSHEY_DUPLEX, size, color, thickness, cv2.LINE_AA)
    return new_images


def add_error_to_images(images, errors, loc='bl', size=0.65, vmin=0., vmax=30.0, thickness=1,
                        precision=3, cl=None, colors=None, cmap=plt.cm.jet):
    assert cl is None or colors is None
    new_images = to_disp_images(images)
    errors = to_numpy(errors)
    if cl is not None:
        colors = [cl for i in range(len(new_images))]
    if colors is None:
        colors = color_map(to_numpy(errors), cmap=cmap, vmin=vmin, vmax=vmax)
        if new_images[0].dtype == np.uint8:
            colors *= 255
    for disp, err, color in zip(new_images, errors, colors):
        pos = get_pos_in_image(loc, size, disp.shape)
        err_str = np.array2string(err, precision=precision, separator=' ', suppress_small=True)
        cv2.putText(disp, err_str, pos, cv2.FONT_HERSHEY_DUPLEX, size, color, thickness, cv2.LINE_AA)
    return new_images


def add_landmarks_to_images(images, landmarks, color=None, radius=1, thickness=-1,
                            connections=None, thickness_contours=1, draw_landmarks=True):

    if landmarks is None:
        return images

    num = min(len(images), len(landmarks))

    new_images = to_disp_images(images[:num])
    landmarks = to_numpy(landmarks[:num])[..., :2]
    default_color = (0, 255, 0)

    if len(landmarks.shape) == 2:
        landmarks = landmarks[np.newaxis]

    for img_id, (disp, lm)  in enumerate(zip(new_images, landmarks)):
        if color is None:
            cl = default_color
        else:
            cl = color

        if connections is not None:
            # for connection in connections:
            #     start_idx = connection[0]
            #     end_idx = connection[1]
            #     cv2.line(disp, lm[start_idx].astype(int), lm[end_idx].astype(int), color, thickness=thickness_contours)

            # connections = dict(connections)
            # st, nd = list(connections.items())[0]
            # contour_ids = [st, nd]
            #
            # for i in range(len(connections)-1):
            #     nd = connections[nd]
            #     contour_ids.append(nd)

            lms2d = lm[..., :2].astype(int)
            contour_ids = [lm1 for (lm1, lm2) in connections]
            contour = [lms2d[i] for i in contour_ids]
            cv2.fillPoly(disp, pts=[np.array(contour)], color=color)

        if draw_landmarks:
            for i in range(0, len(lm)):
                x, y = lm[i].astype(int)
                if x >= 0 and y >= 0:
                    cv2.circle(disp, (x, y), radius=radius, color=cl, thickness=thickness, lineType=cv2.LINE_AA)
    return new_images


def draw_status_bar(text, status_bar_width, status_bar_height, dtype=np.float32, text_size=-1, text_color=(1,1,1)):
    img_status_bar = np.zeros((status_bar_height, status_bar_width, 3), dtype=dtype)
    if text_size <= 0:
        text_size = status_bar_height * 0.020
    cv2.putText(img_status_bar, text, (4,img_status_bar.shape[0]-8), cv2.FONT_HERSHEY_SIMPLEX, text_size, text_color, 1, cv2.LINE_AA)
    return img_status_bar


def overlay_patches(
        dst: np.ndarray,
        src: np.ndarray,
        lms_dst: np.ndarray,
        lms_src: np.ndarray,
        k: int = 5,
        alpha: float = 0.5
) -> np.ndarray:
    h, w, c = src.shape
    half_k = k // 2

    # dst_out = dst.copy()
    dst_out = dst

    for (x_src, y_src), (x_dst, y_dst) in zip(lms_src[:,:2], lms_dst[:,:2]):
        x_src, y_src, x_dst, y_dst = map(int, [x_src, y_src, x_dst, y_dst])

        # Define patch boundaries ensuring they stay within the image
        x1_src, x2_src = max(0, x_src - half_k), min(w, x_src + half_k + 1)
        y1_src, y2_src = max(0, y_src - half_k), min(h, y_src + half_k + 1)
        x1_dst, x2_dst = max(0, x_dst - half_k), min(w, x_dst + half_k + 1)
        y1_dst, y2_dst = max(0, y_dst - half_k), min(h, y_dst + half_k + 1)

        # Get corresponding patch sizes
        patch_src = src[y1_src:y2_src, x1_src:x2_src]
        patch_dst = dst_out[y1_dst:y2_dst, x1_dst:x2_dst]

        # Ensure patches have the same shape before blending (handle border inconsistencies)
        min_h, min_w = min(patch_src.shape[0], patch_dst.shape[0]), min(patch_src.shape[1], patch_dst.shape[1])
        patch_src, patch_dst = patch_src[:min_h, :min_w], patch_dst[:min_h, :min_w]

        # Alpha blending
        blended_patch = (alpha * patch_src + (1 - alpha) * patch_dst).astype(np.uint8)

        # Replace patch in the destination image
        dst_out[y1_dst:y1_dst + min_h, x1_dst:x1_dst + min_w] = blended_patch

        # cv2.circle(dst_out, (x_dst, y_dst), 1, (0, 255, 0), -1)
        # cv2.circle(dst_out, (x_src, y_src), 1, (0, 0, 255), -1)

    return dst_out


def draw_landmarks(img, lms, color=(1, 1, 1), radius=1, thickness=-1):
    _img = img.copy()
    if len(color) == 3:
        colors = np.array([color], dtype=np.float32).repeat(len(lms), axis=0)
    else:
        colors = np.array(color)
    if colors.max() < 1.01:
        colors *= 255
    # colors = colors.astype(int)
    # colors = np.array([int(c * 255) for c in color])

    lms = to_numpy(lms[:, :2])

    if lms.max() < 1.:
        h, w = img.shape[:2]
        lms[:, 0] *= w
        lms[:, 1] *= h

    for pt, cl in zip(lms, colors):
        cv2.circle(_img, pt.astype(int), radius, cl.tolist(), thickness, lineType=cv2.LINE_AA)
    return _img


def draw_projected_landmarks(img, lms3d, camera, color=(1, 1, 1)):
    height, width = img.shape[:2]
    lms2d = backproject_landmarks(lms3d, camera, width, height)
    return draw_landmarks(img, lms2d, color=color)


def draw_landmark_polygon(img, lms, connections, color=(1, 1, 1), thickness=1):
    _img = to_numpy(img).copy()
    lms = to_numpy(lms[:, :2])
    for connection in connections:
        start_idx = connection[0]
        end_idx = connection[1]
        cv2.line(_img, lms[start_idx].astype(int), lms[end_idx].astype(int), color, thickness)
    return _img
