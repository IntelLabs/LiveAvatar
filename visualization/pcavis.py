import numpy as np
import torch
import matplotlib.pyplot as plt
from utils.nn import to_numpy
from sklearn.decomposition import PCA
import sklearn


class PCAVis():
    def __init__(self, segment=False, bg_thresh=0.5):
        self.pca = None
        self.pca_fg = None
        self.segment = segment
        self.bg_thresh = bg_thresh

    @staticmethod
    def _plot_components(pca_features):
        """ visualize PCA components for finding a proper threshold 3 histograms for 3 components"""
        plt.subplot(2, 2, 1)
        plt.hist(pca_features[:, 0])
        plt.subplot(2, 2, 2)
        plt.hist(pca_features[:, 1])
        plt.subplot(2, 2, 3)
        plt.hist(pca_features[:, 2])
        plt.show()
        plt.close()

    @staticmethod
    def _show_component(pca_features, comp_id,  H, W):
        for i in range(len(pca_features)):
            plt.subplot(2, 2, i + 1)
            plt.imshow(pca_features[i * H * W: (i + 1) * H * W, comp_id].reshape(H, W))
        plt.show()
        plt.close()

    def fit(self, features: np.ndarray | torch.Tensor):
        features = to_numpy(features)

        N, D, H, W = features.shape
        features = features.transpose(0, 2, 3, 1).reshape(-1, D)

        # print(f"Fitting PCA for background segmentation...")
        self.pca = PCA(n_components=3)
        self.pca.fit(features)

        if self.segment:
            pca_features = self.pca.transform(features)
            pca_features = self._minmax_scale(pca_features)

            # segment/seperate the backgound and foreground using the first component
            pca_features_bg = pca_features[:, 0] < self.bg_thresh  # from first histogram
            pca_features_fg = ~pca_features_bg

            # print(f"Fitting PCA for only foreground patches...")
            self.pca_fg = PCA(n_components=3)
            self.pca_fg.fit(features[pca_features_fg])

    @staticmethod
    def _minmax_scale(features):
        features = sklearn.preprocessing.minmax_scale(features)
        return features

    def transform(self, features: np.ndarray | torch.Tensor):
        features = to_numpy(features)

        input_ndim = len(features.shape)
        if input_ndim == 3:
            features = features[np.newaxis]

        if self.pca is None:
            self.fit(features)

        N, D, H, W = features.shape

        if D == 9:  # or D == 32:
            features = features[:, :3]
            D = 3

        if D == 3:
            pca_features = features.transpose(0, 2, 3, 1)

            f = pca_features
            for i in range(3):
                f[..., i] = (f[...,i] - f[...,i].min()) / (f[..., i].max() - f[...,i].min())
            return (255 * f).astype(np.uint8)

        features = features.transpose(0, 2, 3, 1).reshape(-1, D)

        pca_features = self.pca.transform(features)
        pca_features = self._minmax_scale(pca_features)

        if self.segment:
            # segment/seperate the backgound and foreground using the first component
            pca_features_bg = pca_features[:, 0] < self.bg_thresh
            pca_features_fg = ~pca_features_bg

            pca_features_left = self.pca_fg.transform(features[pca_features_fg])
            pca_features_left = self._minmax_scale(pca_features_left)

            pca_features_rgb = pca_features.copy()
            pca_features_rgb[pca_features_bg] = 0
            pca_features_rgb[pca_features_fg] = pca_features_left
        else:
            pca_features_rgb = pca_features

        # reshaping to numpy image format
        pca_features_rgb = pca_features_rgb.reshape(N, H, W, 3)

        if input_ndim == 3:
            pca_features_rgb = pca_features_rgb[0]

        return (pca_features_rgb * 255).astype(np.uint8)
