import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarkerResult
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

import itertools
import torch

from visualization.vis import show_image
from visualization import vis


LM68_RIGID_IDS = list(itertools.chain(*[
    [37, 40],  # right eye corners
    [43, 46],  # left eye corners
    range(28, 36),  # nose
    [49, 55],  # mouth corners
]))


def lm478_to_lm68(lms: np.ndarray):
    if len(lms.shape) == 2:
        return convert_landmarks_mediapipe_to_dlib(lms)
    elif len(lms.shape) == 3:
        return  np.array([convert_landmarks_mediapipe_to_dlib(l) for l in lms])
    else:
        raise ValueError("landmarks must be of shape [N, 478, D] or [478, D]")


def tensor_lm478_to_lm68(lms: torch.Tensor):
    if len(lms.shape) == 3:
        return lms[:, mp2dlib_correspondence].mean(dim=2)
    else:
        raise ValueError("landmarks must be of shape [N, 478, D]")


def flip_landmarks(lms, mirror_ids):
    lms_new = lms.copy()
    lms_new[:, 0] -= 0.5
    lms_new[:, 0] *= -1
    lms_new[:, 0] += 0.5
    return lms_new[mirror_ids]


def face_to_numpy(det_result: FaceLandmarkerResult) -> np.ndarray:
    N = len(det_result.face_landmarks[0])
    keypoints = np.zeros((N, 5))
    for i in range(N):
        keypoints[i, 0] = det_result.face_landmarks[0][i].x
        keypoints[i, 1] = det_result.face_landmarks[0][i].y
        keypoints[i, 2] = det_result.face_landmarks[0][i].z
        keypoints[i, 3] = det_result.face_landmarks[0][i].visibility
        keypoints[i, 4] = det_result.face_landmarks[0][i].presence
    return keypoints


def numpy_to_face(det_result, keypoints):
    N = len(keypoints)
    for i in range(N):
        det_result.face_landmarks[0][i].x = keypoints[i, 0]
        det_result.face_landmarks[0][i].y = keypoints[i, 1]
        det_result.face_landmarks[0][i].z = keypoints[i, 2]
        det_result.face_landmarks[0][i].visibility = keypoints[i, 3]
        det_result.face_landmarks[0][i].presence = keypoints[i, 4]
    return det_result


def to_mp_image(img_rgb: np.ndarray):
    return mp.Image(
        image_format=mp.ImageFormat.SRGB,
        # data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        data=img_rgb.copy()
    )


def draw_landmarks_on_face(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return annotated_image



VisionRunningMode = mp.tasks.vision.RunningMode

class LandmarkDetector():
    def __init__(self, asset_dir, running_mode='image', flip_input: bool = True, min_confidence=0.5,
                 gpu: bool = True):
        self._mode = VisionRunningMode.IMAGE if running_mode == 'image' else VisionRunningMode.VIDEO
        self._options = vision.FaceLandmarkerOptions(
            running_mode=self._mode,
            base_options=python.BaseOptions(
                model_asset_path=os.path.join(asset_dir, 'face_landmarker_v2_with_blendshapes.task'),
                delegate=python.BaseOptions.Delegate.GPU if gpu else python.BaseOptions.Delegate.CPU,
                # delegate=python.BaseOptions.Delegate.CPU,
            ),
            num_faces=1,
            min_face_detection_confidence=min_confidence,
            min_face_presence_confidence=min_confidence
        )
        self._face_detector = None
        self._mirror_ids = np.loadtxt(os.path.join(asset_dir, "mediapipe_mirror_ids_obj.txt")).astype(int)
        self._with_flip = flip_input
        self.reset()

    def reset(self):
        if self._mode == VisionRunningMode.VIDEO:
            if self._face_detector is not None:
                self._face_detector.close()
            self._face_detector = vision.FaceLandmarker.create_from_options(self._options)
        else:
            if self._face_detector is None:
                self._face_detector = vision.FaceLandmarker.create_from_options(self._options)

    def detect(
            self,
            image_rgb: np.ndarray,
            frame_timestamp_ms: int | None = None,
            show: bool = True
    ) -> np.ndarray | None:
        mp_image = to_mp_image(image_rgb)

        if self._mode == VisionRunningMode.IMAGE:
            face_detection_result = self._face_detector.detect(mp_image)
        elif self._mode == VisionRunningMode.VIDEO:
            face_detection_result = self._face_detector.detect_for_video(mp_image, frame_timestamp_ms)
        else:
            raise ValueError()

        if not face_detection_result.face_landmarks:
            return None

        landmarks = face_to_numpy(face_detection_result)
        # score = np.mean([bs.score for bs in face_detection_result.face_blendshapes[0]])

        if self._with_flip:
            mp_image_flip = to_mp_image(np.fliplr(image_rgb))
            face_detection_result_flip = self._face_detector.detect(mp_image_flip)
            landmarks_flip = face_to_numpy(face_detection_result_flip)
            landmarks_unflip = flip_landmarks(landmarks_flip, self._mirror_ids)
            landmarks = 0.5 * landmarks + 0.5 * landmarks_unflip

        if show:
            annotated_image = draw_landmarks_on_face(mp_image.numpy_view(), face_detection_result)
            cv2.imshow("result_face", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

            # annotated_image_flip = draw_landmarks_on_face(mp_image_flip.numpy_view(), face_detection_result_flip)
            # cv2.imshow("result_face flip", cv2.cvtColor(annotated_image_flip, cv2.COLOR_RGB2BGR))
            #
            # face_detection_result_unflip = numpy_to_face(face_detection_result, landmarks_unflip)
            # annotated_image = draw_landmarks_on_face(annotated_image, face_detection_result_unflip)
            # cv2.imshow("result_face with flip", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

            cv2.waitKey(1)

        return landmarks[:, :3]


class LandmarkDetectorFAN():
    def __init__(self, flip_input: bool = True):
        import face_alignment
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=flip_input, device='cuda')

    def detect(self, image_rgb, show: bool = True):
        preds, scores, _ = self.fa.get_landmarks_from_image(image_rgb, return_landmark_score=True)
        # print(scores)
        if preds is None:
            return None, None
        preds = preds[0]
        h, w = image_rgb.shape[:2]
        preds[:, 0] /= w
        preds[:, 1] /= h
        preds[:, 2] /= h
        # if show:
        #     vis_image = vis.add_landmarks_to_images(image_rgb, preds, color=(0, 255, 0))
        #     show_image("landmarks FAN", vis_image, wait=5)
        return preds, scores



mp2dlib_correspondence = [

    ## Face Contour
    [127],  # 1
    [234],  # 2
    [93],  # 3
    [132, 58],  # 4
    [58, 172],  # 5
    [136],  # 6
    [150],  # 7
    [176],  # 8
    [152],  # 9
    [400],  # 10
    [379],  # 11
    [365],  # 12
    [397, 288],  # 13
    [361],  # 14
    [323],  # 15
    [454],  # 16
    [356],  # 17

    ## Right Brow
    [70],  # 18
    [63],  # 19
    [105],  # 20
    [66],  # 21
    [107],  # 22

    ## Left Brow
    [336],  # 23
    [296],  # 24
    [334],  # 25
    [293],  # 26
    [300],  # 27

    ## Nose
    [168, 6],  # 28
    [197, 195],  # 29
    [5],  # 30
    [4],  # 31
    [75],  # 32
    [97],  # 33
    [2],  # 34
    [326],  # 35
    [305],  # 36

    ## Right Eye
    [33],  # 37
    [160],  # 38
    [158],  # 39
    [133],  # 40
    [153],  # 41
    [144],  # 42

    ## Left Eye
    [362],  # 43
    [385],  # 44
    [387],  # 45
    [263],  # 46
    [373],  # 47
    [380],  # 48

    ## Upper Lip Contour Top
    [61],  # 49
    [39],  # 50
    [37],  # 51
    [0],  # 52
    [267],  # 53
    [269],  # 54
    [291],  # 55

    ## Lower Lip Contour Bottom
    [321],  # 56
    [314],  # 57
    [17],  # 58
    [84],  # 59
    [91],  # 60

    ## Upper Lip Contour Bottom
    [78],  # 61
    [82],  # 62
    [13],  # 63
    [312],  # 64
    [308],  # 65

    ## Lower Lip Contour Top
    [317],  # 66
    [14],  # 67
    [87],  # 68
]


for ri in range(68):
    if len(mp2dlib_correspondence[ri]) == 1:
        idx = mp2dlib_correspondence[ri][0]
        mp2dlib_correspondence[ri] = [idx, idx]


def convert_landmarks_mediapipe_to_dlib(lmks_mp: np.array):
    """
    Convert the 478 Mediapipe dense landmarks to
    the 68 Dlib sparse landmarks
    input:
        - lmks_mp: Mediapipe landmarks, [478, 2] or [478, 3]
    return:
        - lmks_mp2dlib: Converted Dlib landmarks, [68, 2] or [68, 3]
    """
    return lmks_mp[mp2dlib_correspondence].mean(axis=1)