from .goldberg import ImageSignature
import numpy as np


class TransformerSignature(ImageSignature):

    def __init__(self, transformer, n=9, crop_percentiles=(5, 95), P=None, diagonal_neighbors=True,
      identical_tolerance=2/255., n_levels=2, fix_ratio=False):
        super().__init__(n, crop_percentiles, P, diagonal_neighbors,
                         identical_tolerance, n_levels, fix_ratio)
        self.transformer = transformer

    def generate_signature(self, path_or_image, bytestream=False):
        im_array = self.preprocess_image(path_or_image, handle_mpo=self.handle_mpo, bytestream=bytestream)
        return self.transformer.transform(np.expand_dims(im_array, 0))

    def expand_to_rgb(self, img):
        if len(img.shape) > 2:
            return img
        else:
            return np.dstack([img] * 3)
